from __future__ import absolute_import, division, print_function
import numpy as np
import os
import cv2
from tqdm import tqdm
from numpy import linalg as LA

from pwcnet import ModelPWCNet

pwc_net = ModelPWCNet()


def getFlowBD(img0, img1, windowName=''):
    FLOW_THRES = 2               # threshold to accept a flow by bi-directional matching
    h, w = img0.shape[:2]

    img_pairs = [(img0, img1), (img1, img0)]
    flow01, flow10 = pwc_net.predict_from_img_pairs(img_pairs, batch_size=2)

    flow01_filtered = np.empty_like(flow01)
    flow01_filtered[:] = np.nan
    for v0 in range(h):
        for u0 in range(w):
            fu01, fv01 = flow01[v0, u0, :]
            u1, v1 = u0+fu01, v0+fv01
            u1i, v1i = int(u1+0.5), int(v1+0.5)
            if 0 <= v1i < h and 0 <= u1i < w:
                fu10, fv10 = flow10[v1i, u1i, :]
                du, dv = u1+fu10-u0, v1+fv10-v0
                if (du*du+dv*dv) < FLOW_THRES:  # bi-directional filtering
                    flow01_filtered[v0, u0, 0] = flow01[v0, u0, 0]
                    flow01_filtered[v0, u0, 1] = flow01[v0, u0, 1]
    return flow01_filtered


def getRay(cam, uv):
    ray_uv = np.ones(3, dtype=np.float32)
    ray_uv[0] = (uv[0]-cam[2]) / cam[0]
    ray_uv[1] = (uv[1]-cam[3]) / cam[1]
    return np.expand_dims(ray_uv / LA.norm(ray_uv), -1)


def depthFromTriangulation(cam_ref, cam_cur, T_cur_ref, uv_ref, uv_cur):
    R_cur_ref = T_cur_ref[0:3, 0:3]
    t_cur_ref = T_cur_ref[0:3, 3]
    ray_uv_ref = getRay(cam_ref, uv_ref)
    ray_uv_cur = getRay(cam_cur, uv_cur)

    A = np.hstack((np.matmul(R_cur_ref, ray_uv_ref), ray_uv_cur))
    AtA = np.matmul(A.T, A)
    if LA.det(AtA) < 1e-5:
        return -1
    depth2 = - np.matmul(np.matmul(LA.inv(AtA), A.T), t_cur_ref)
    depth = np.fabs(depth2[0])

    return depth*ray_uv_ref[-1]


def calculateCurDepth(cam0, img0, cam1, img1, T_cam0_v1, v1_lut):
    h, w = img0.shape[:2]

    depth1 = np.empty([h, w])
    depth1[:] = np.nan

    flow10 = getFlowBD(img1, img0, 'Match')
    for v1 in range(h):
        for u1 in range(w):
            fu, fv = flow10[v1, u1, :]
            if not np.isnan(fu):
                uv0 = [u1+fu, v1+fv]
                depth1[v1, u1] = depthFromTriangulation(
                    cam1, cam0, T_cam0_v1[v1_lut[v1, u1]], [u1, v1], uv0)

    return depth1


def getDepth(save_path):
    img0_path = save_path+"cam0/images/"
    img1_path = save_path+"cam1/images/"
    depth1_path = save_path+"cam1/depth/"
    if not os.path.exists(depth1_path):
        os.makedirs(depth1_path)

    # Load cameras
    cam0 = np.load(save_path+"cam0/camera.npy")
    cam1 = np.load(save_path+"cam1/camera.npy")

    # Load poses
    T_cam0_v1 = np.load(save_path+"poses_cam0_v1.npy")
    v1_lut = np.load(save_path+"cam1/v1_lut.npy")
    img_count = T_cam0_v1.shape[0]

    # Get depth
    for i in tqdm(range(img_count)):
        img0 = cv2.imread('{}{}.png'.format(img0_path, i))
        img1 = cv2.imread('{}{}.png'.format(img1_path, i))
        depth1 = calculateCurDepth(
            cam0, img0, cam1, img1, T_cam0_v1[i], v1_lut)
        fname = os.path.join(depth1_path, str(i))
        np.save(fname, depth1)
