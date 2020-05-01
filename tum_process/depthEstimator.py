from __future__ import absolute_import,division,print_function
import numpy as np
import os
import cv2
import shutil
from tqdm import tqdm
from numpy import linalg as LA
from scipy.spatial.transform import Rotation

from flowExtractor import getFlowBD,getFlowEp

def getRay(cam,uv):
    ray_uv = np.ones(3,dtype=np.float32)
    ray_uv[0] = (uv[0]-cam[2]) / cam[0]
    ray_uv[1] = (uv[1]-cam[3]) / cam[1]
    return np.expand_dims(ray_uv / LA.norm(ray_uv),-1)

def depthFromTriangulation(cam_ref,cam_cur,T_cur_ref,uv_ref,uv_cur):
    R_cur_ref = T_cur_ref[0:3,0:3]
    t_cur_ref = T_cur_ref[0:3,3]
    ray_uv_ref = getRay(cam_ref,uv_ref)
    ray_uv_cur = getRay(cam_cur,uv_cur)

    A = np.hstack((np.matmul(R_cur_ref,ray_uv_ref),ray_uv_cur))
    AtA = np.matmul(A.T,A)
    if LA.det(AtA) < 1e-5:
        return -1
    depth2 = - np.matmul(np.matmul(LA.inv(AtA),A.T),t_cur_ref)
    depth = np.fabs(depth2[0])

    return depth*ray_uv_ref[-1]

def calculateCurDepth(cam0,img0,cam1,img1,T_cam0_v1):
    h,w = img0.shape[:2]

    depth1 = np.empty([h,w])
    depth1[:] = np.nan    
    
    flow10 = getFlowBD(img1,img0,'Rolling Stereo Match')
    for v1 in range(h):
        for u1 in range(w):
            fu,fv = flow10[v1,u1,:]
            if not np.isnan(fu):
                uv0 = [u1+fu,v1+fv]
                v0 = int(v1+fv+0.5)
                if(0<=v0<h):
                    depth1[v1,u1] = depthFromTriangulation(cam1,cam0,T_cam0_v1[v1],[u1,v1],uv0)

    return depth1  

def getDepth(save_dir):
    depth1_dir = save_dir+"cam1/depth/"
    if os.path.exists(depth1_dir): shutil.rmtree(depth1_dir)
    os.makedirs(depth1_dir)

    # Load images
    img0_dir = save_dir+"cam0/images/"
    img1_dir = save_dir+"cam1/images/"
    img_count = len(os.listdir(img0_dir))

    # Load poses
    T_cam0_v1 = np.load(save_dir+"poses_cam0_v1.npy")

    # Load cameras
    cam0 = np.load(save_dir+"cam0/camera.npy")
    cam1 = np.load(save_dir+"cam1/camera.npy")

    # Get depth 
    for i in tqdm(range(img_count)):
        img0 = cv2.imread('{}{}.png'.format(img0_dir,i))
        img1 = cv2.imread('{}{}.png'.format(img1_dir,i))
        depth1 = calculateCurDepth(cam0,img0,cam1,img1,T_cam0_v1[i])
        fname = os.path.join(depth1_dir,str(i))
        np.save(fname,depth1)
        