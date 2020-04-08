from __future__ import absolute_import,division,print_function
import numpy as np
import os
import cv2
from tqdm import tqdm
from numpy import linalg as LA
from scipy.spatial.transform import Rotation
from depth_filter import Seed,transferPoint,filterDepthByFlow,updateSeed

from flow_extractor import getFlowBD,getFlowEp

def calculateCurDepth(cam0,prv_depth,imgs0,cur_i,nearby_frames,T_rel_0,cam1,img1,T_cam0_v1):
    h,w = prv_depth.shape

    cur_depth = np.empty_like(prv_depth)
    cur_depth[:] = np.nan    

    # Project depth 
    depth_min = np.inf
    if cur_i>0:
        flow_cur2prv = getFlowBD(imgs0[cur_i],imgs0[cur_i-1],'Consecutive frames GS')
        PROJ_FLOW_DIFF_THRES = 2        # Threshold to accept depth
        for cv in range(h):
            for cu in range(w):
                [fu,fv] = flow_cur2prv[cv,cu,:]
                if np.isnan(fu): continue
                pu,pv = cu+fu,cv+fv
                [cu_p,cv_p,cd] = transferPoint(pu,pv,prv_depth,T_rel_0[cur_i,cur_i-1],cam0,cam0)
                if np.isnan(cu_p): continue
                [du,dv] = [cu_p-cu,cv_p-cv]
                if (du*du+dv*dv)<PROJ_FLOW_DIFF_THRES:
                    cur_depth[cv,cu] = cd
                    if depth_min>cd: depth_min = cd
    
    # Initialize seed for pixels without depth
    seeds = []
    flow01 = getFlowBD(imgs0[cur_i],img1,'Rolling Stereo Match')
    for v0 in range(h):
        for u0 in range(w):
            if not np.isnan(cur_depth[v0,u0]): continue # already have depth

            cur_seed = Seed((u0,v0))
            # try to initialize depth by rolling stereo
            fu,fv = flow01[v0,u0,:]
            if not np.isnan(fu):
                uv1 = (u0+fu,v0+fv)
                v1 = int(v0+fv+0.5)
                if(0<=v1<h):
                    cur_seed = updateSeed(cam0,cam1,cur_seed,uv1,T_cam0_v1[v1])
                    cd = 1.0/cur_seed.mu
                    if depth_min>cd: depth_min = cd
            seeds.append(cur_seed)
    # set z_range = 1/depth_min
    if depth_min==np.inf:
        depth_min = 1.0
    z_range = 1.0/depth_min
    for i in range(len(seeds)): seeds[i].z_range = z_range

    # Update seeds using nearby_frames
    for fi in nearby_frames:
        T_cur_fi = T_rel_0[cur_i,fi]
        flow_cur2fi = getFlowEp(imgs0[cur_i],imgs0[fi],cam0,T_cur_fi,cam0,'Depth Filter Flow')
        seeds = filterDepthByFlow(cam0,seeds,flow_cur2fi,T_cur_fi)

    # update depth
    SEED_CONVERGE_SIGMA2_THRESH = 200.0
    for seed in seeds:
        if seed.sigma2*seed.sigma2<seed.z_range/SEED_CONVERGE_SIGMA2_THRESH: 
            [u,v] = seed.uv  
            cur_depth[v,u] = 1.0/seed.mu
    # print(np.sum(~np.isnan(cur_depth))/h/w)
    return cur_depth  

def getDepth(save_dir):
    depth_dir = save_dir+"cam0/depth/"
    if os.path.exists(depth_dir): shutil.rmtree(depth_dir)
    os.makedirs(depth_dir)

    # Load images
    img0_dir = save_dir+"cam0/images/"
    img1_dir = save_dir+"cam1/images/"
    img_count = len(os.listdir(img0_dir))
    imgs0,imgs1 = [],[]
    for i in range(img_count):
        img0 = cv2.imread('{}{}.png'.format(img0_dir,i))
        img1 = cv2.imread('{}{}.png'.format(img1_dir,i))
        imgs0.append(img0)
        imgs1.append(img1)
    h,w = imgs0[0].shape[:2]

    # Load poses
    T_w = np.load(save_dir+"poses_cam0.npy")
    T_w_inv = np.empty_like(T_w)
    for i in range(img_count):
        T_w_inv[i] = LA.inv(T_w[i])
    T_rel_0 = np.empty((img_count,img_count,4,4))
    for i in range(img_count):
        for j in range(img_count):
            T_i_w = T_w_inv[i]
            T_w_j = T_w[j]
            T_rel_0[i,j] = np.matmul(T_i_w,T_w_j)
    T_cam0_v1 = np.load(save_dir+"poses_v1.npy")

    # Depth filter parameters
    df_dist = 0.1 
    df_angle = 3/180*np.pi
    df_frames = 5
    px_noise = 1.0
    cam0 = np.load(save_dir+"cam0/camera.npy")
    cam1 = np.load(save_dir+"cam1/camera.npy")
    px_error_angle = np.arctan(px_noise/(2.0*np.fabs(cam0[0])))*2.0 # px_error_angle
    cam0 = (cam0[0],cam0[1],cam0[2],cam0[3],px_error_angle)

    # Get depth by projection and depth filtering
    prv_depth = np.empty((h,w))  
    prv_depth[:] = np.nan 
    for cur_i in tqdm(range(img_count)):
        # Find good nearby frames to calculate depth
        nearby_frames = []
        s=e=cur_i
        while e<img_count and LA.norm(T_rel_0[e,s][0:3,3])<df_dist: e+=1
        while len(nearby_frames)<df_frames:
            while e<img_count and LA.norm(T_rel_0[e,s][0:3,3])<df_dist \
            and LA.norm(Rotation.from_matrix(T_rel_0[e,s][0:3,0:3]).as_rotvec())<df_angle:
                e+=1
            if e>=img_count:
                break
            nearby_frames.append(e)
            s=e

        # Current depth
        cur_depth = calculateCurDepth(cam0,prv_depth,imgs0,cur_i,nearby_frames,T_rel_0,cam1,imgs1[cur_i],T_cam0_v1[cur_i])
        fname = os.path.join(depth_dir,str(cur_i))
        np.save(fname,cur_depth)
        
        prv_depth = cur_depth