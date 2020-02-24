from __future__ import absolute_import, division, print_function
import numpy as np
import cv2
import pathlib
import os
import csv
import rospy
import yaml
from numpy import linalg as LA
from tqdm import tqdm
import shutil
from scipy.spatial.transform import Rotation as sciRot
from scipy.spatial.transform import Slerp
from numba import jit, cuda

from pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from depth_filter import castRay, getRay, updateSeed

################################# helper: get optical flow using PWC-Net ##########################################
def drawFlow(img, flow):
    h, w = img.shape[:2]
    flow_vis = img
    edge = cv2.Canny(img, 40, 80)
    for v in range(h):
        for u in range(w):
            if edge[v, u]: 
                fu,fv = flow[v,u].T
                if(fu*fu+fv*fv>1):
                    cv2.line(flow_vis, (u, v), (np.int32(u+fu+0.5), np.int32(v+fv+0.5)), (0, 255, 0))

    return flow_vis

@jit
def filterFlow(flow01, flow10):
    FLOW_THRES = 2
    h,w = flow01.shape[:2]
    flow01_filtered = np.empty(flow01.shape)
    flow01_filtered[:] = np.NaN
    for v0 in range(h):
        for u0 in range(w):
            fu01,fv01 = flow01[v0,u0,:]
            u1,v1 = int(u0+fu01+0.5),int(v0+fv01+0.5)
            if 0<=v1<h and 0<=u1<w:
                fu10,fv10 = flow10[v1,u1,:]
                u0l,v0l = u1+fu10,v1+fv10
                dist = (u0l-u0)*(u0l-u0)+(v0l-v0)*(v0l-v0)
                if dist<FLOW_THRES:
                    flow01_filtered[v0,u0,0] = flow01[v0,u0,0]
                    flow01_filtered[v0,u0,1] = flow01[v0,u0,1]
    return flow01_filtered

pwc_net = ModelPWCNet()
def getFlow(img0Path, img1Path, windowName):
    img0 = cv2.imread(img0Path)
    img1 = cv2.imread(img1Path)
    img_pairs = [(img0, img1), (img1, img0)]
    flow01, flow10 = pwc_net.predict_from_img_pairs(img_pairs, batch_size=2)
    flow01_filtered = filterFlow(flow01, flow10)
    
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.imshow(windowName, drawFlow(img0, flow01_filtered))
    cv2.waitKey(1)  

    return flow01_filtered
################################# helper: get optical flow using PWC-Net ##########################################

######################################### get pose for each image #################################################
def getT(t_q):
    T = np.identity(4, dtype=np.float32)
    T[0:3,0:3] = sciRot.from_quat([t_q[np.ix_([4,5,6,3])]]).as_matrix()[0] # convert to qx qy qz qw for from_quat
    T[0:3,3] = t_q[0:3].T
    return T
    
def poseInterpolate(t_q_0, t_q_1, ns0, ns1, ns_cur):
    assert(ns0<=ns_cur<=ns1)
    t0 = t_q_0[0:3]
    t1 = t_q_1[0:3]
    R01 = sciRot.from_quat([t_q_0[np.ix_([4,5,6,3])], t_q_1[np.ix_([4,5,6,3])]])    # convert to qx qy qz qw for from_quat
    t_cur = ((ns1-ns_cur)*t0 + (ns_cur-ns0)*t1) / (ns1-ns0)
    q_cur = Slerp([ns0, ns1], R01)(ns_cur).as_quat()
    t_q_cur = np.hstack([t_cur, q_cur[np.ix_([3,0,1,2])]])  # convert back to qw qx qy qz
    # print(t_q_0)
    # print(t_q_cur)
    # print(t_q_1)
    # print('\n\n\n')
    return t_q_cur

def tumGetPoses(data_dir, save_dir):
    # Read calibration file
    with open(data_dir+"camchain.yaml") as file:
        camchain = yaml.load(file, Loader=yaml.FullLoader)
        T_cam0_imu = np.matrix(camchain['cam0']['T_cam_imu'])
    T2rectified = np.identity(4,dtype=np.float32)
    T2rectified[0:3,0:3]=np.load(save_dir+"cam0/R2rectified.npy")
    T_cam0_imu = T2rectified * T_cam0_imu

    # image name/time_ns
    ns = np.array(list(np.loadtxt(open(data_dir+'cam0/times.txt'), delimiter=" ")))
    ns = ns[:,0].astype(np.int)

    # pose ground truth
    gt_reader = csv.reader(open(data_dir+'gt_imu.csv'), delimiter=",")
    next(gt_reader)
    gt_ns_t_q = np.array(list(gt_reader))
    gt_ns = gt_ns_t_q[:,0].astype(np.int)
    gt_t_q = gt_ns_t_q[:,1:8].astype(np.float32)
    gt_i = 0

    poses = np.zeros((len(ns), 7), dtype=np.float32)
    for i in tqdm(range(len(ns))):
        # Interpolate pose
        while gt_i<len(gt_ns) and gt_ns[gt_i]<ns[i]:
            gt_i += 1
        
        if gt_i==0:
            cur_t_q = gt_t_q[0]
        elif gt_i==len(gt_ns):
            cur_t_q = gt_t_q[-1]
        else:
            cur_t_q = poseInterpolate(gt_t_q[gt_i-1], gt_t_q[gt_i], gt_ns[gt_i-1], gt_ns[gt_i], ns[i])
        
        # transform to cam0 frame
        T_w_imu = getT(cur_t_q)
        T_w_cam0 = np.matmul(T_w_imu, LA.inv(T_cam0_imu))
        t_cam0 = [T_w_cam0[0,3], T_w_cam0[1,3], T_w_cam0[2,3]]
        q_cam0 = sciRot.from_matrix(T_w_cam0[0:3,0:3]).as_quat()
        t_q_cam0 = np.hstack([t_cam0, q_cam0[np.ix_([3,0,1,2])]])  # convert back to qw qx qy qz
        # print(T_w_imu)
        # print(T_w_cam0)
        poses[i,:] = t_q_cam0

    pose_path = os.path.join(save_dir,"cam0/poses.npy")
    np.save(pose_path, poses)
######################################### get pose for each image #################################################

################################################ stereo rectify ###################################################
def saveCamera(fileName, P):
    fxfycxcytx = np.array([P[0,0],P[1,1],P[0,2],P[1,2],P[0,3]])
    np.save(fileName, fxfycxcytx)


def tumStereoRectify(data_dir, save_dir, resolution):
    img0_dir = save_dir+"cam0/images/"
    img1_dir = save_dir+"cam1/images/"
    if os.path.exists(img0_dir): shutil.rmtree(img0_dir)
    if os.path.exists(img1_dir): shutil.rmtree(img1_dir)
    os.makedirs(img0_dir)
    os.makedirs(img1_dir)

    # Read original calibration file
    with open(data_dir+"camchain.yaml") as file:
        camchain = yaml.load(file, Loader=yaml.FullLoader)

        imageSize = tuple(camchain['cam0']['resolution'])
        cam0_intrinsics = camchain['cam0']['intrinsics']
        K0 = np.matrix([[cam0_intrinsics[0], 0, cam0_intrinsics[2]], 
                    [0, cam0_intrinsics[1], cam0_intrinsics[3]],
                    [0,0,1]])
        D0 = np.array(camchain['cam0']['distortion_coeffs'])

        cam1_intrinsics = camchain['cam1']['intrinsics']
        K1 = np.matrix([[cam1_intrinsics[0], 0, cam1_intrinsics[2]], 
                    [0, cam1_intrinsics[1], cam1_intrinsics[3]],
                    [0,0,1]])
        D1 = np.array(camchain['cam1']['distortion_coeffs'])

        T01 = np.matrix(camchain['cam1']['T_cn_cnm1'])
        R = T01[np.ix_([0,1,2], [0,1,2])]
        tvec = T01[np.ix_([0,1,2], [3])]

    # Fisheye stere0 rectify
    R0,R1,P0,P1,Q = cv2.fisheye.stereoRectify(K0,D0,K1,D1,imageSize,R,tvec,0,newImageSize=resolution)
    map00, map01 = cv2.fisheye.initUndistortRectifyMap(K0, D0, R0, P0, resolution, cv2.CV_32F)
    map10, map11 = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, resolution, cv2.CV_32F)
    map0, map1 = [map00, map10], [map01, map11]
    np.save(save_dir+"cam0/R2rectified.npy", R0)
    saveCamera(save_dir+"cam0/camera.npy", P0)
    saveCamera(save_dir+"cam1/camera.npy", P1)

    img_names = sorted(os.listdir(data_dir+"cam0/images/"))
    for img_i in tqdm(range(len(img_names))):
        for i in range(2):
            img_name = img_names[img_i]
            img = cv2.imread(f'{data_dir}cam{i}/images/{img_name}')
            # Remap images 
            img_rect = cv2.remap(img, map0[i], map1[i], cv2.INTER_LINEAR)
            save_path = f'{save_dir}cam{i}/images/{img_i}.png'
            cv2.imwrite(save_path, img_rect)
################################################ stereo rectify ###################################################

####################################### get depth using depth filter ##############################################
# @jit
def updateSeedsByFlow(cam,seeds,flow_cur2fi,T_cur_fi):
    for i in range(len(seeds)):
        [cu,cv] = seeds[i][0]
        if np.isnan(flow_cur2fi[cv,cu,0]):
            continue
        fu,fv = flow_cur2fi[cv,cu,:]
        updatedSeed = updateSeed(cam,seeds[i],(cu+fu,cv+fv), T_cur_fi)
        seeds[i] = updatedSeed
    return seeds

def calculateDepth(cam,prv_depth,img_dir,cur_i,frames,T_pose):
    h,w = prv_depth.shape[:2]
    depth = -np.ones((h,w), dtype=np.float32)

    # Project prv_depth to current frame
    depth_min = np.inf
    if cur_i>0:
        T_cur_prv = T_pose[cur_i,cur_i-1,:,:]
        for pv in range(h):
            for pu in range(w):
                if prv_depth[pv,pu]>0:
                    X_prv = getRay(cam, (pu,pv))*prv_depth[pv,pu]
                    X_cur = np.matmul(T_cur_prv[0:3,0:3],X_prv)+np.expand_dims(T_cur_prv[0:3,3],-1)
                    [cu, cv] = castRay(cam, X_cur)
                    if 0<=cu<w and 0<=cv<h:
                        depth[cv,cu] = LA.norm(X_cur)
                        if depth_min>depth[cv,cu]:
                            depth_min = depth[cv,cu]
    if depth_min>100:
        depth_min = 1
    
    # Initialize seed for pixels without prv_depth projection
    seeds = []
    a = 10
    b = 10
    mu = -1.0
    z_range = 1.0/depth_min
    sigma2 = -1.0
    for v in range(h):
        for u in range(w):
            if depth[v,u]<0:
                seeds.append(((u,v),a,b,mu,z_range,sigma2))

    # Update seeds using frames
    for fi in frames:
        T_cur_fi = T_pose[cur_i,fi,:,:]
        flow_cur2fi = getFlow(f'{img_dir}{cur_i}.png', f'{img_dir}{fi}.png', 'Depth Filter Flow')
        print([cur_i, fi])
        seeds = updateSeedsByFlow(cam,seeds,flow_cur2fi,T_cur_fi)

    # update depth
    for seed in seeds:
        # print(seed)
        # if seed[5]<0.01: #TODO
        if seed[3]>0: #TODO
            [u,v] = seed[0]
            depth[v,u] = 1.0 / seed[3]

    d_s=[]
    for v in range(h):
        for u in range(w):
            if depth[v,u]>0:
                d_s.append(depth[v,u])
    print([np.mean(d_s), len(d_s)/h/w])
    return depth

def getDepth(save_dir, img_sz):
    # Parameters for Depth Filter
    df_dist = 0.1 
    df_frames = 5

    img_dir = save_dir+"cam0/images/"
    depth_dir = save_dir+"cam0/depth/"
    if os.path.exists(depth_dir): shutil.rmtree(depth_dir)
    os.makedirs(depth_dir)
    img_count = len(os.listdir(img_dir))

    # Load poses
    print('Loading poses...')
    poses = np.load(save_dir+"cam0/poses.npy")
    T_w = np.empty((poses.shape[0], 4, 4), dtype=np.float32)
    T_w_inv = np.empty((poses.shape[0], 4, 4), dtype=np.float32)
    for i in range(poses.shape[0]):
        T_w[i,:,:] = getT(poses[i])
        T_w_inv[i,:,:] = LA.inv(T_w[i,:,:])
    T_pose = np.empty((poses.shape[0], poses.shape[0], 4, 4), dtype=np.float32)
    for i in range(poses.shape[0]):
        for j in range(poses.shape[0]):
            T_i_w = T_w_inv[i,:,:]
            T_w_j = T_w[j,:,:]
            T_pose[i,j,:,:] = np.matmul(T_i_w, T_w_j)

    # Create depth filter
    cam0 = np.load(save_dir+"cam0/camera.npy")
    px_noise = 1.0
    px_error_angle = np.arctan(px_noise/(2.0*np.fabs(cam0[0])))*2.0 # px_error_angle
    cam = (cam0[0],cam0[1],cam0[2],cam0[3], px_error_angle)

    # Create empty depth map for the first frame
    depth = -np.ones((img_sz[1],img_sz[0]), np.float32)
    for i in tqdm(range(img_count)):
        # Find good frames to calculate depth
        frames = []
        s=e=i
        while len(frames)<df_frames:
            while e<img_count and LA.norm(T_pose[e,s,::][0:3,3])<df_dist:
                e+=1
            frames.append(e)
            s=e

        depth = calculateDepth(cam,depth,img_dir,i,frames,T_pose)

        # Save current depth
        fname = os.path.join(depth_dir, f'{i}')
        np.save(fname,depth)

####################################### get depth using depth filter ##############################################

###################################### get unrolling ground-truth flow #############################################
@jit
def getFlowsRs2Gen(flow01,depth0,cam0,ftx):
    h,w = flow01.shape[:2]
    flows_rs2gen = np.empty(flow01.shape)
    flows_rs2gen[:] = np.NaN
    # Project from cam0 to cam1 
    for v0 in range(h):
        for u0 in range(w):
            if depth0[v0,u0]>0:
                X0 = getRay(cam0, (u0,v0))*depth0[v0,u0]
                z0 = X0[2,0]
                [u1d,v1d] = [u0+ftx/z0,v0]
                if 0<=u1d<w:
                    [uf,vf] = flow01[v0,u0].T
                    if not np.isnan(uf):
                        [u1f,v1f] = [int(u0+uf+0.5), int(v0+vf+0.5)]
                        if 0<=u1f<w and 0<=v1f<h:
                            print([z0,uf, ftx/z0])
                            flows_rs2gen[v1f,u1f,0] = u1d-(u0+uf)
                            flows_rs2gen[v1f,u1f,1] = v1d-(v0+vf)
    return flows_rs2gen

def getUnrollingFlow(save_dir):
    img0_dir = save_dir+"cam0/images/"
    depth0_dir = save_dir+"cam0/depth/"
    img1_dir = save_dir+"cam1/images/"
    flows_rs2gen_dir = save_dir+"cam1/flows_rs2gen/"
    if os.path.exists(flows_rs2gen_dir): shutil.rmtree(flows_rs2gen_dir)
    os.makedirs(flows_rs2gen_dir)

    cam0 = np.load(save_dir+"cam0/camera.npy")
    cam1 = np.load(save_dir+"cam1/camera.npy")
    ftx = cam1[4]

    img_count = len(os.listdir(img0_dir))
    for i in tqdm(range(img_count)):
        flow01 = getFlow(f'{img0_dir}{i}.png', f'{img1_dir}{i}.png', 'Stereo Flow')
        depth0 = np.load(f'{depth0_dir}{i}.npy')
        flows_rs2gen = getFlowsRs2Gen(flow01,depth0,cam0,ftx)

        fname = os.path.join(flows_rs2gen_dir, f'{i}')
        np.save(fname,flows_rs2gen)
###################################### get unrolling ground-truth flow #############################################

if __name__=="__main__":
    resolution = (320, 256)
    data_dir = "/home/jiawei/Workspace/data/dataset-seq1/dso/"
    save_dir = os.path.join(pathlib.Path(__file__).resolve().parent.parent.absolute(), "data/seq1/")
    # if os.path.exists(save_dir): shutil.rmtree(save_dir)
    # os.makedirs(save_dir)

    # print('Stereo Rectifying...')
    # tumStereoRectify(data_dir, save_dir, resolution)
    # print('Getting Pose...')
    # tumGetPoses(data_dir, save_dir)
    print('Getting Depth...')
    getDepth(save_dir, resolution)
    # print('Getting Unrolling Flow...')
    # getUnrollingFlow(save_dir)