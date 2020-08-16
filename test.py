from __future__ import print_function, division
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation, RotationSpline
from scipy import interpolate
import pandas as pd

from model.depthnet import DepthNet
from model.anchornet import AnchorNet
from data_loader import dataLoader


def projectPoint(ua, va, da, cam_a, T_b_a, cam_b):
    Xa = np.ones([3, 1], dtype=np.float32)
    Xa[0] = (ua-cam_a[2]) / cam_a[0]
    Xa[1] = (va-cam_a[3]) / cam_a[1]
    Xa = Xa*da
    Xb = np.matmul(T_b_a[0:3, 0:3], Xa)+np.expand_dims(T_b_a[0:3, 3], -1)
    ub = Xb[0, 0]/Xb[2, 0]*cam_b[0] + cam_b[2]
    vb = Xb[1, 0]/Xb[2, 0]*cam_b[1] + cam_b[3]
    return [ub, vb]


def getGS2RSFlow(depth_rs, cam1, anchors_t_r, rot_weight):
    num_anchor = len(anchors_t_r)
    h, w = depth_rs.shape[:2]
    flow_gs2rs = np.empty([h, w, 2], dtype=np.float32)
    flow_gs2rs[:] = np.nan

    tm = np.arange(num_anchor+1) / num_anchor
    ts, rs = [[0, 0, 0]], [[0, 0, 0]]
    for i in range(num_anchor):
        ts.append(list(anchors_t_r[i][:3]))
        rs.append(list(anchors_t_r[i][3:]/rot_weight))
    t_spline = interpolate.CubicSpline(tm, ts)
    R_spline = RotationSpline(tm, Rotation.from_rotvec(rs))

    # Project from rs to gsc
    for v1rs in range(h):
        tm = v1rs/(h-1)
        T_gs_rs = np.identity(4)
        T_gs_rs[0:3, 0:3] = R_spline(tm).as_matrix()
        T_gs_rs[0:3, 3] = t_spline(tm)
        for u1rs in range(w):
            if np.isnan(depth_rs[v1rs, u1rs]):
                continue

            [u1gs, v1gs] = projectPoint(
                u1rs, v1rs, depth_rs[v1rs, u1rs], cam1, T_gs_rs, cam1)
            if not np.isnan(u1gs):
                u1gsi, v1gsi = int(u1gs+0.5), int(v1gs+0.5)
                if 0 <= u1gsi < w and 0 <= v1gsi < h:
                    flow_gs2rs[v1gsi, u1gsi, 0] = u1rs-u1gs
                    flow_gs2rs[v1gsi, u1gsi, 1] = v1rs-v1gs

    return flow_gs2rs


def rectifyImgByFlow(img_input, flow_pred):
    img = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
    h, w = img.shape[:2]
    indy, indx = np.indices((h, w), dtype=np.float32)
    flow_df_u = pd.DataFrame(flow_pred[:, :, 0])
    flow_df_v = pd.DataFrame(flow_pred[:, :, 1])
    flow_interp = np.empty_like(flow_pred)
    flow_interp[:, :, 0] = flow_df_u.interpolate(
        method='linear', limit_direction='forward', axis=0)
    flow_interp[:, :, 1] = flow_df_v.interpolate(
        method='linear', limit_direction='forward', axis=0)
    map_x = indx.reshape(h, w).astype(np.float32) + flow_interp[:, :, 0]
    map_y = indy.reshape(h, w).astype(np.float32) + flow_interp[:, :, 1]
    img_rectified = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return img_rectified


def markImgResults(img, pred_dist, zero_dist):
    text_color = (0, 255, 0) if zero_dist > pred_dist else (0, 0, 255)
    img_text = cv2.putText(img, "{:.2f}".format(
        pred_dist), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    return img_text


# read num_anchor and test_seq from command line
parser = argparse.ArgumentParser()
parser.add_argument('--num_anchor', help='Number of anchors to predict')
parser.add_argument('--test_seq', default=2, help='Test sequence')
args = parser.parse_args()
test_seq = int(args.test_seq)
num_anchor = int(args.num_anchor)
print('Number of anchors to predict: {}'.format(num_anchor))
print('Test sequence: {}'.format(test_seq))

# parameters
rot_weight = 10

# load data
data_loader = dataLoader([test_seq])
imgs = data_loader.loadTestingImg()
depths = data_loader.loadTestingDepth()
flows = data_loader.loadTestingFlow()
cam1 = np.load(os.path.join(os.getcwd(), "data/seq1/cam1/camera.npy"))

# load model
depthnet = DepthNet(data_loader.getImgShape())
depthnet.model.load_weights(os.path.join(
    os.getcwd(), "model/checkpoints/model_depth.hdf5"))
anchornet = AnchorNet(data_loader.getImgShape(), num_anchor)
anchornet.model.load_weights(os.path.join(os.getcwd(
), 'model/checkpoints/{}/model_anchor{}.hdf5'.format(rot_weight, num_anchor)))

# path to save results
save_dir = os.path.join(
    os.getcwd(), "test_results/{}/{}/".format(rot_weight, num_anchor))
if not os.path.exists(save_dir+'images/'):
    os.makedirs(save_dir+'images/')

win_dv = 0
err_dv = []
for i in tqdm(range(len(imgs))):
    img_input = np.expand_dims(imgs[i], 0)  # (1, h, w, 1)

    depth_pred = depthnet.model.predict(img_input)[0]
    anchor_pred = anchornet.model.predict(img_input)[0]
    anchor_pred = np.reshape(anchor_pred, (-1, 6))

    flow_pred = getGS2RSFlow(depth_pred, cam1, anchor_pred, rot_weight)
    img_rectified = rectifyImgByFlow(img_input[0], flow_pred)
    cv2.imwrite('{}images/{}.png'.format(save_dir, i), img_rectified)

    flow_gt = flows[i]
    pred_dist = np.nanmean(
        np.sqrt(np.sum(np.square(flow_gt-flow_pred), axis=-1)))
    zero_dist = np.nanmean(np.sqrt(np.sum(np.square(flow_gt), axis=-1)))
    err_dv.append(pred_dist)
    win_dv += (1 if zero_dist > pred_dist else 0)
    # img_mark = markImgResults(img_rectified, pred_dist, zero_dist)

print('Improved Ratio: {}'.format(win_dv/len(imgs)))
print('EPE err: {}'.format(np.mean(err_dv)))
np.save(save_dir+'errs.npy', np.array(err_dv))
