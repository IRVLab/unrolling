from __future__ import print_function, division
import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from unrollnet.unrollnet import UnrollNet
from evaluation.depthvelnet.depthnet import DepthNet
from evaluation.depthvelnet.velocitynet import VelocityNet
from data_loader import dataLoader

def projectPoint(ua,va,da,cam_a,T_b_a,cam_b):
    Xa = np.ones([3,1],dtype=np.float32)
    Xa[0] = (ua-cam_a[2]) / cam_a[0]
    Xa[1] = (va-cam_a[3]) / cam_a[1]
    Xa = Xa*da
    Xb = np.matmul(T_b_a[0:3,0:3],Xa)+np.expand_dims(T_b_a[0:3,3],-1)
    ub = Xb[0,0]/Xb[2,0]*cam_b[0] + cam_b[2]
    vb = Xb[1,0]/Xb[2,0]*cam_b[1] + cam_b[3]
    return [ub,vb]

def getGS2RSFlowCV(depth_rs,cam1,vel_t_r,ns_per_v):
    h,w = depth_rs.shape[:2]
    flow_gs2rs = np.empty([h,w,2], dtype=np.float32)
    flow_gs2rs[:] = np.nan

    # Project from rs to gs
    for v1rs in range(h):
        tm = v1rs*ns_per_v/1e9
        T_gs_rs = np.identity(4)
        T_gs_rs[0:3,0:3] = Rotation.from_rotvec(tm*vel_t_r[3:]).as_matrix()
        T_gs_rs[0:3,3] = tm*vel_t_r[:3]
        for u1rs in range(w): 
            if np.isnan(depth_rs[v1rs,u1rs]): continue
            
            [u1gs,v1gs] = projectPoint(u1rs,v1rs,depth_rs[v1rs,u1rs],cam1,T_gs_rs,cam1)
            if not np.isnan(u1gs): 
                u1gsi,v1gsi = int(u1gs+0.5),int(v1gs+0.5)
                if 0<=u1gsi<w and 0<=v1gsi<h: 
                    flow_gs2rs[v1gsi,u1gsi,0] = u1rs-u1gs
                    flow_gs2rs[v1gsi,u1gsi,1] = v1rs-v1gs

    return flow_gs2rs

def rectifyImgByFlow(img, flow):
    h, w = img.shape[:2]
    indy, indx = np.indices((h, w), dtype=np.float32)
    map_x = indx.reshape(h, w).astype(np.float32) + flow[:,:,0]
    map_y = indy.reshape(h, w).astype(np.float32) + flow[:,:,1]
    rectified_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return rectified_img

def process(img_input, flow_gt, zero_dist, flow_pred, pred_str):
    img_pred = rectifyImgByFlow(img_input, flow_pred)
    pred_dist = np.nanmean(np.sqrt(np.sum(np.square(flow_gt-flow_pred), axis=-1)))
    text_color = (0,255,0) if zero_dist > pred_dist else (0,0,255)
    img_pred_text = cv2.putText(img_pred, "{}{:.2f}".format(pred_str,pred_dist), (10,240), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2) 
    return [img_pred_text, pred_dist]

# dataset and experiment directories
data_loader = dataLoader() 
imgs, flows = data_loader.loadTestingUnroll()
depths, vels = data_loader.loadTestingDepthVelocity()

unrollnet = UnrollNet(data_loader.getImgShape())
unrollnet.model.load_weights(os.path.join(os.getcwd(), "checkpoints/model_unroll.hdf5"))

depthnet = DepthNet(data_loader.getImgShape())
depthnet.model.load_weights(os.path.join(os.getcwd(), "checkpoints/model_depth.hdf5"))

velocitynet = VelocityNet(data_loader.getImgShape())
velocitynet.model.load_weights(os.path.join(os.getcwd(), "checkpoints/model_velocity.hdf5"))

save_dir = os.path.join(os.getcwd(), "test_results/")
if not os.path.exists(save_dir+'images/'): os.makedirs(save_dir+'images/')

cam1 = np.load(os.path.join(os.getcwd(), "data/seq1/cam1/camera.npy"))
resolution = (320,256)
ns_per_v = 29.4737*1023/(resolution[1]-1)*1000

## training pipeline
wins_ur, wins_dv, wins_cg = 0, 0, 0
errs_ur, errs_dv, errs_cg = [], [], []
for i in tqdm(range(len(imgs))):
    img_input = np.expand_dims(imgs[i], 0) # (1, h, w, 1)
    flow_gt = flows[i]
    zero_dist = np.nanmean(np.sqrt(np.sum(np.square(flow_gt), axis=-1)))

    flow_pred_ur = unrollnet.model.predict(img_input)[0]
    depth_pred = depthnet.model.predict(img_input)[0]
    vel_pred = velocitynet.model.predict(img_input)[0,0,0]
    flow_pred_dv = getGS2RSFlowCV(depth_pred,cam1,vel_pred,ns_per_v)
    # flow_pred_cg = getGS2RSFlowCV(depths[i],cam1,vels[i][0][0],ns_per_v)
    
    img_input = cv2.cvtColor(img_input[0], cv2.COLOR_GRAY2RGB)
    # img_gt = rectifyImgByFlow(img_input, flow_gt)

    [img_pred_text_ur, pred_dist_ur] = process(img_input, flow_gt, zero_dist, flow_pred_ur, 'UnrollNet: ')
    [img_pred_text_dv, pred_dist_dv] = process(img_input, flow_gt, zero_dist, flow_pred_dv, 'DepthVelNet: ')
    # [img_pred_text_cg, pred_dist_cg] = process(img_input, flow_gt, zero_dist, flow_pred_cg, 'CV_GT: ')
    
    # img_input_text = cv2.putText(img_input, 'Input: {:.2f}'.format(zero_dist), (10,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) 
    # img_gt_text = cv2.putText(img_gt, 'Ground-Truth', (10,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) 
    # concat_img = cv2.hconcat([img_input_text,img_gt_text,img_pred_text_ur,img_pred_text_dv,img_pred_text_cg])
    # cv2.imwrite('{}images/{}.png'.format(save_dir,i), concat_img)

    errs_ur.append(pred_dist_ur)
    errs_dv.append(pred_dist_dv)
    # errs_cg.append(pred_dist_cg)
    wins_ur += (1 if zero_dist > pred_dist_ur else 0)
    wins_dv += (1 if zero_dist > pred_dist_dv else 0)
    # wins_cg += (1 if zero_dist > pred_dist_cg else 0)

np.save(save_dir+'errs_unrollnet.npy', np.array(errs_ur))
np.save(save_dir+'errs_depthvelnet.npy', np.array(errs_dv))
# np.save(save_dir+'errs_cv_gt.npy', np.array(errs_cg))
print ('Improved Ratio:')
print ('UnrollNet: {}'.format(wins_ur/len(imgs)))
print ('DepthVelNet: {}'.format(wins_dv/len(imgs)))
# print ('CV_GT: {}'.format(wins_cg/len(imgs)))
print ('EPE:')
print('UnrollNet err: {}'.format(np.mean(errs_ur)))
print('DepthVelNet err: {}'.format(np.mean(errs_dv)))
# print('CV_GT err: {}'.format(np.mean(errs_cg)))
