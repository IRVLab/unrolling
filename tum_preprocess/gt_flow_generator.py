from __future__ import absolute_import,division,print_function
import numpy as np
import os
import cv2
import shutil
from tqdm import tqdm

from depth_filter import transferPoint
from flow_extractor import getFlowBD

def getGS2RSFlow(flow10,depth0,cam0,cam1,T_1_0):
    h,w = flow10.shape[:2]
    flow_gs2rs = np.empty_like(flow10)
    flow_gs2rs[:] = np.nan

    # Project from cam0 to cam1 
    for v1 in range(h):
        for u1 in range(w): 
            uf,vf = flow10[v1,u1,:]
            if np.isnan(uf): continue
            u0,v0 = u1+uf,v1+vf
            [u1gs,_,_] = transferPoint(u0,v0,depth0,T_1_0,cam0,cam1)
            v1gs = v0    # v1gs=v0 due to stereo
            if np.isnan(u1gs): continue
            u1gsi,v1gsi = int(u1gs+0.5),int(v1gs+0.5)
            if 0<=u1gsi<w and 0<=v1gsi<h: 
                flow_gs2rs[v1gsi,u1gsi,0] = u1-u1gs
                flow_gs2rs[v1gsi,u1gsi,1] = v1-v1gs
                            
    return flow_gs2rs

def getGS2RSFlows(save_dir):
    img0_dir = save_dir+"cam0/images/"
    depth0_dir = save_dir+"cam0/depth/"
    img1_dir = save_dir+"cam1/images/"
    flows_gs2rs_dir = save_dir+"cam1/flows_gs2rs/"
    if os.path.exists(flows_gs2rs_dir): shutil.rmtree(flows_gs2rs_dir)
    os.makedirs(flows_gs2rs_dir)

    cam0 = np.load(save_dir+"cam0/camera.npy")
    cam1 = np.load(save_dir+"cam1/camera.npy")
    T_1_0 = np.identity(4)
    T_1_0[0,3] = cam1[4]

    img_count = len(os.listdir(img0_dir))
    for i in tqdm(range(img_count)):
        img0 = cv2.imread('{}{}.png'.format(img0_dir,i))
        img1 = cv2.imread('{}{}.png'.format(img1_dir,i))
        flow10 = getFlowBD(img1,img0,'Stereo Flow')
        depth0 = np.load('{}{}.npy'.format(depth0_dir,i))
        flow_gs2rs = getGS2RSFlow(flow10,depth0,cam0,cam1,T_1_0)
        gs2rs_name = os.path.join(flows_gs2rs_dir,str(i))
        np.save(gs2rs_name,flow_gs2rs)