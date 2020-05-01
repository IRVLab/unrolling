from __future__ import absolute_import,division,print_function
import numpy as np
import os
import cv2
import shutil
from tqdm import tqdm
from numpy import linalg as LA
from scipy.spatial.transform import Rotation

def projectPoint(ua,va,da,cam_a,T_b_a,cam_b):
    Xa = np.ones([3,1],dtype=np.float32)
    Xa[0] = (ua-cam_a[2]) / cam_a[0]
    Xa[1] = (va-cam_a[3]) / cam_a[1]
    Xa = Xa*da
    Xb = np.matmul(T_b_a[0:3,0:3],Xa)+np.expand_dims(T_b_a[0:3,3],-1)
    ub = Xb[0,0]/Xb[2,0]*cam_b[0] + cam_b[2]
    vb = Xb[1,0]/Xb[2,0]*cam_b[1] + cam_b[3]
    return [ub,vb]

def getGS2RSFlow(depth_rs,cam1,T_cam0_v1):
    h,w = depth_rs.shape[:2]
    flow_gs2rs = np.empty([h,w,2], dtype=np.float32)
    flow_gs2rs[:] = np.nan

    # Project from rs to gs
    for v1rs in range(h):
        T_gs_rs = np.matmul(LA.inv(T_cam0_v1[0]),T_cam0_v1[v1rs])
        for u1rs in range(w): 
            if np.isnan(depth_rs[v1rs,u1rs]): continue

            [u1gs,v1gs] = projectPoint(u1rs,v1rs,depth_rs[v1rs,u1rs],cam1,T_gs_rs,cam1)
            if not np.isnan(u1gs): 
                u1gsi,v1gsi = int(u1gs+0.5),int(v1gs+0.5)
                if 0<=u1gsi<w and 0<=v1gsi<h: 
                    flow_gs2rs[v1gsi,u1gsi,0] = u1rs-u1gs
                    flow_gs2rs[v1gsi,u1gsi,1] = v1rs-v1gs

    return flow_gs2rs

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

def getGS2RSFlows(save_dir,ns_per_v):
    depth1_dir = save_dir+"cam1/depth/"
    cam1 = np.load(save_dir+"cam1/camera.npy")
    flows_gs2rs_dir = save_dir+"cam1/flows_gs2rs/"
    if os.path.exists(flows_gs2rs_dir): shutil.rmtree(flows_gs2rs_dir)
    os.makedirs(flows_gs2rs_dir)
    flows_gs2rs_cv_dir = save_dir+"cam1/flows_gs2rs_cv/"
    if os.path.exists(flows_gs2rs_cv_dir): shutil.rmtree(flows_gs2rs_cv_dir)
    os.makedirs(flows_gs2rs_cv_dir)

    # Load poses
    T_cam0_v1 = np.load(save_dir+"poses_cam0_v1.npy")
    const_vel_cam1 = np.load(save_dir+"cam1/vel_t_r.npy")

    img_count = len(os.listdir(depth1_dir))
    cv_diff = []
    for i in tqdm(range(img_count)):
        depth1 = np.load('{}{}.npy'.format(depth1_dir,i))
        flow_gs2rs = getGS2RSFlow(depth1,cam1,T_cam0_v1[i])
        flow_gs2rs_cv = getGS2RSFlowCV(depth1,cam1,const_vel_cam1[i],ns_per_v)

        gs2rs_name = os.path.join(flows_gs2rs_dir,str(i))
        np.save(gs2rs_name,flow_gs2rs)
        gs2rs_cv_name = os.path.join(flows_gs2rs_cv_dir,str(i))
        np.save(gs2rs_cv_name,flow_gs2rs_cv)

        cv_diff.append(np.nanmean(np.square(flow_gs2rs-flow_gs2rs_cv)))

    np.save(save_dir+"cv_diff.npy",cv_diff)