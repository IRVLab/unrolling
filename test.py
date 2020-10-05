from __future__ import print_function, division
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation, RotationSpline
from scipy import interpolate
import pandas as pd
from numpy import linalg as LA

from model.depthnet import DepthNet
from model.anchornet import AnchorNet
from model.unrollnet import UnrollNet
from data_loader import dataLoader


def getGS2RSFlow(depth_rs, cam, anchors_t_r):
    num_anchor = len(anchors_t_r)
    h, w = depth_rs.shape[:2]
    flow_gs2rs = np.empty([h, w, 2], dtype=np.float32)
    flow_gs2rs[:] = np.nan

    tm = np.arange(num_anchor+1) / num_anchor
    ts, rs = [[0, 0, 0]], [[0, 0, 0]]
    for i in range(num_anchor):
        ts.append(list(anchors_t_r[i][:3]))
        rs.append(list(anchors_t_r[i][3:]))
    t_spline = interpolate.CubicSpline(tm, ts)
    R_spline = RotationSpline(tm, Rotation.from_rotvec(rs))

    K = np.array([[cam[0], 0, cam[2]], [0, cam[1], cam[3]], [0, 0, 1]])
    K_i = LA.inv(K)

    # Project from rs to gs
    for v_rs in range(h):
        tm = v_rs/(h-1)
        KRK_i = np.matmul(np.matmul(K, R_spline(tm).as_matrix()), K_i)
        Kt = np.matmul(K, t_spline(tm))
        for u_rs in range(w):
            if np.isnan(depth_rs[v_rs, u_rs]):
                continue

            p_gs = depth_rs[v_rs, u_rs] * \
                np.matmul(KRK_i, np.array([u_rs, v_rs, 1])) + Kt
            u_gs, v_gs = p_gs[0] / p_gs[2], p_gs[1] / p_gs[2]
            if not np.isnan(u_gs):
                u_gsi, v_gsi = int(u_gs+0.5), int(v_gs+0.5)
                if 0 <= u_gsi < w and 0 <= v_gsi < h:
                    flow_gs2rs[v_gsi, u_gsi, 0] = u_rs-u_gs
                    flow_gs2rs[v_gsi, u_gsi, 1] = v_rs-v_gs

    return flow_gs2rs


def rectifyImgByFlow(img, flow):
    h, w = img.shape[:2]
    indy, indx = np.indices((h, w), dtype=np.float32)
    flow_u = pd.DataFrame(flow[:, :, 0])
    flow_v = pd.DataFrame(flow[:, :, 1])
    flow_interp = np.empty_like(flow)
    flow_interp[:, :, 0] = flow_u.interpolate(
        method='linear', limit_direction='forward', axis=0)
    flow_interp[:, :, 1] = flow_v.interpolate(
        method='linear', limit_direction='forward', axis=0)
    map_x = indx.reshape(h, w).astype(np.float32) + flow_interp[:, :, 0]
    map_y = indy.reshape(h, w).astype(np.float32) + flow_interp[:, :, 1]
    img_rectified = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return img_rectified


# read num_anchor from command line
parser = argparse.ArgumentParser()
parser.add_argument('--num_anchor', help='Number of anchors to predict')
parser.add_argument(
    '--depth_gt', help='Whether to use ground-truth depth', default=0)
parser.add_argument(
    '--rectify_img', help='Whether to rectify images', default=0)
args = parser.parse_args()
num_anchor = int(args.num_anchor)
print('Number of anchors to predict: {}'.format(num_anchor))
use_depth_gt = True if int(args.depth_gt) > 0 else False
print('Use {} depth'.format('ground-truth' if use_depth_gt else 'predicted'))
rectify_img = True if int(args.rectify_img) > 0 else False
print('Rectify image: {}'.format(rectify_img))

# load data
data_loader = dataLoader()
imgs = data_loader.loadTestingImg()
total_count = len(imgs)
if use_depth_gt:
    depths = data_loader.loadTestingDepth()
flows = data_loader.loadTestingFlow()
# anchors = data_loader.loadTestingAnchor(num_anchor)
cam1 = np.load(os.path.join(os.getcwd(), "data/seq1/cam1/camera.npy"))

# load model
if num_anchor == 0:
    unrollnet = UnrollNet(data_loader.getImgShape())
    unrollnet.model.load_weights(os.path.join(os.getcwd(
    ), 'checkpoints/model_unroll.hdf5'))
elif num_anchor > 0:
    depthnet = DepthNet(data_loader.getImgShape())
    depthnet.model.load_weights(os.path.join(
        os.getcwd(), "checkpoints/model_depth.hdf5"))
    anchornet = AnchorNet(data_loader.getImgShape(), num_anchor)
    anchornet.model.load_weights(os.path.join(os.getcwd(
    ), 'checkpoints/model_anchor{}.hdf5'.format(num_anchor)))

# path to save results
save_dir = os.path.join(os.getcwd(
), "test_results/{}/".format('depth_gt' if use_depth_gt else 'depth_pred'))
img_dir = save_dir+'images{}/'.format(num_anchor)
if rectify_img and not os.path.exists(img_dir):
    os.makedirs(img_dir)

wins = 0
errs = []
imgs_res = []
rs_intense = []
for i in tqdm(range(total_count)):
    img_input_rgb = np.expand_dims(imgs[i], 0)  # (1, h, w, 3)
    img_input = np.expand_dims(img_input_rgb[:, :, :, 0], -1)  # (1, h, w, 1)
    flow_gt = flows[i]

    if num_anchor == 0:
        flow_pred = unrollnet.model.predict(img_input)[0]
    elif num_anchor > 0:
        if use_depth_gt:
            depth_pred = depths[i]
        else:
            depth_pred = depthnet.model.predict(img_input)[0]
        if data_loader.inverse_depth:
            depth_pred = np.reciprocal(depth_pred)
        anchor_pred = anchornet.model.predict(img_input_rgb)[0]
        anchors_pred_t_r = np.reshape(anchor_pred, (-1, 6))
        anchors_pred_t_r[:, :3] = anchors_pred_t_r[:, :3] / \
            data_loader.trans_weight
        flow_pred = getGS2RSFlow(depth_pred, cam1, anchors_pred_t_r)
    else:
        flow_pred = np.zeros_like(flow_gt)

    pred_dist = np.nanmean(
        np.sqrt(np.sum(np.square(flow_gt-flow_pred), axis=-1)))
    zero_dist = np.nanmean(np.sqrt(np.sum(np.square(flow_gt), axis=-1)))
    errs.append(pred_dist)
    wins += (1 if zero_dist > pred_dist else 0)

    if rectify_img:
        img_rgb = 255*img_input_rgb[0]
        img_rectified = rectifyImgByFlow(img_rgb, flow_pred)
        img_rectified_gt = rectifyImgByFlow(img_rgb, flow_gt)
        img_res = cv2.hconcat([img_rgb, img_rectified, img_rectified_gt])
        imgs_res.append(img_res)
        rs_intense.append(zero_dist)

print('Improved Ratio: {}'.format(wins/len(errs)))
print('EPE errs:        {}'.format(np.mean(errs)))
np.save(save_dir+'errss{}.npy'.format(num_anchor), np.array(errs))

if rectify_img:
    # intensive rs image goes first
    rs_sorted_idx = np.argsort(rs_intense)
    for i in range(total_count):
        cv2.imwrite(img_dir+'{}.png'.format(i),
                    imgs_res[rs_sorted_idx[total_count-1-i]])
