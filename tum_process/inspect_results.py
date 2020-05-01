from __future__ import absolute_import,division,print_function
import numpy as np
import os
import cv2
import shutil
from tqdm import tqdm
from numpy import linalg as LA

def recitifyImg(img_rs,flow_gs2rs):
    h,w = img_rs.shape[:2]
    indy, indx = np.indices((h, w), dtype=np.float32)
    map_x = indx.reshape(h, w).astype(np.float32) + flow_gs2rs[:,:,0]
    map_y = indy.reshape(h, w).astype(np.float32) + flow_gs2rs[:,:,1]
    rectified_img = cv2.remap(img_rs, map_x, map_y, cv2.INTER_LINEAR)
    rectified_img = cv2.putText(rectified_img, str(-np.nanmean(flow_gs2rs[:,:,0])), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) 
    rectified_img = cv2.putText(rectified_img, str(-np.nanmean(flow_gs2rs[:,:,1])), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2) 
    return rectified_img

def inspectResults(save_dir):
    # Load images
    img1_dir = save_dir+"cam1/images/"
    depth1_dir = save_dir+"cam1/depth/"
    flows_gs2rs_dir = save_dir+"cam1/flows_gs2rs/"
    flows_gs2rs_cv_dir = save_dir+"cam1/flows_gs2rs_cv/"
    const_vel_cam1 = np.load(save_dir+"cam1/vel_t_r.npy")
    img_count = len(os.listdir(img1_dir))

    for i in tqdm(range(img_count)):
        print(const_vel_cam1[i])
        img1 = cv2.imread('{}{}.png'.format(img1_dir,i),0)
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        depth1 = np.load('{}{}.npy'.format(depth1_dir,i))
        flow_gs2rs = np.load('{}{}.npy'.format(flows_gs2rs_dir,i))
        flow_gs2rs_cv = np.load('{}{}.npy'.format(flows_gs2rs_cv_dir,i))
        h, w = img1.shape[:2]
    
        dep1_jet = depth1/7*255
        dep1_jet = cv2.applyColorMap(dep1_jet.astype(np.uint8), cv2.COLORMAP_JET)

        rectified_img1 = recitifyImg(img1,flow_gs2rs)
        rectified_img1_cv = recitifyImg(img1,flow_gs2rs_cv)

        img_dep_rect = cv2.hconcat([img1,dep1_jet,rectified_img1,rectified_img1_cv])
        cv2.imshow('Result',img_dep_rect)
        cv2.waitKey()

if __name__=="__main__":
    seqs = ['1','2','3','4','5','6','7','8','9','10']
    for seq in seqs:
        save_dir = os.path.join(os.getcwd(),'../data/seq{}/'.format(seq))
        inspectResults(save_dir)