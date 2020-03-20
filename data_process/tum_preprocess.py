from __future__ import absolute_import,division,print_function
import numpy as np
import cv2
import os
import csv
import yaml
from numpy import linalg as LA
from tqdm import tqdm
import shutil
from scipy.spatial.transform import Rotation,Slerp
from numba import jit

from geometry import getEH,getEpipolarLine
from flow_extractor import getFlowBD,getFlowEp
from depth_filter import transferPoint,filterDepthByFlow

################################################ stereo rectify ###################################################
def saveCamera(fileName,P):
    fxfycxcytx = np.array([P[0,0],P[1,1],P[0,2],P[1,2],P[0,3]/P[0,0]])
    np.save(fileName,fxfycxcytx)

def stereoRectify(data_dir,save_dir,resolution):
    print ('Stereo Rectifying...')

    img0_dir = save_dir+"cam0/images/"
    img1_dir = save_dir+"cam1/images/"
    if os.path.exists(img0_dir): shutil.rmtree(img0_dir)
    if os.path.exists(img1_dir): shutil.rmtree(img1_dir)
    os.makedirs(img0_dir)
    os.makedirs(img1_dir)

    # Read original calibration file
    with open(data_dir+"camchain.yaml") as file:
        camchain = yaml.load(file,Loader=yaml.FullLoader)

        imageSize = tuple(camchain['cam0']['resolution'])
        cam0_intrinsics = camchain['cam0']['intrinsics']
        K0 = np.matrix([[cam0_intrinsics[0],0,cam0_intrinsics[2]],
                    [0,cam0_intrinsics[1],cam0_intrinsics[3]],
                    [0,0,1]])
        D0 = np.array(camchain['cam0']['distortion_coeffs'])

        cam1_intrinsics = camchain['cam1']['intrinsics']
        K1 = np.matrix([[cam1_intrinsics[0],0,cam1_intrinsics[2]],
                    [0,cam1_intrinsics[1],cam1_intrinsics[3]],
                    [0,0,1]])
        D1 = np.array(camchain['cam1']['distortion_coeffs'])

        T01 = np.matrix(camchain['cam1']['T_cn_cnm1'])
        R = T01[np.ix_([0,1,2],[0,1,2])]
        tvec = T01[np.ix_([0,1,2],[3])]

    # Fisheye stere0 rectify
    R0,R1,P0,P1,Q = cv2.fisheye.stereoRectify(K0,D0,K1,D1,imageSize,R,tvec,0,newImageSize=resolution)
    map00,map01 = cv2.fisheye.initUndistortRectifyMap(K0,D0,R0,P0,resolution,cv2.CV_32F)
    map10,map11 = cv2.fisheye.initUndistortRectifyMap(K1,D1,R1,P1,resolution,cv2.CV_32F)
    map0,map1 = [map00,map10],[map01,map11]

    T_cam0_imu = np.matrix(camchain['cam0']['T_cam_imu'])
    T2rectified = np.identity(4)
    T2rectified[0:3,0:3]=R0
    T_imu_cam0 = LA.inv(T2rectified*T_cam0_imu)
    np.save(save_dir+"cam0/T_imu_cam0.npy",T_imu_cam0)

    saveCamera(save_dir+"cam0/camera.npy",P0)
    saveCamera(save_dir+"cam1/camera.npy",P1)

    # Remap images 
    img_names = sorted(os.listdir(data_dir+"cam0/images/"))
    for img_i in tqdm(range(len(img_names))):
        for i in range(2):
            img_name = img_names[img_i]
            img = cv2.imread('{}cam{}/images/{}'.format(data_dir,i,img_name))
            img_rect = cv2.remap(img,map0[i],map1[i],cv2.INTER_LINEAR)
            save_path = '{}cam{}/images/{}.png'.format(save_dir,i,img_i)
            cv2.imwrite(save_path,img_rect)
################################################ stereo rectify ###################################################

######################################### get pose for each image #################################################
def getT(t_q):
    T = np.identity(4)
    T[0:3,0:3] = Rotation.from_quat([t_q[np.ix_([4,5,6,3])]]).as_matrix()[0] # convert to qx qy qz qw for from_quat
    T[0:3,3] = t_q[0:3].T
    return T
    
def getInterpolatedT(t_q_0,t_q_1,ns0,ns1,ns_cur):
    assert(ns0<=ns_cur<=ns1)
    t0 = t_q_0[0:3]
    t1 = t_q_1[0:3]
    R01 = Rotation.from_quat([t_q_0[np.ix_([4,5,6,3])],t_q_1[np.ix_([4,5,6,3])]])  # convert to qx qy qz qw 
    t_cur = ((ns1-ns_cur)*t0 + (ns_cur-ns0)*t1) / (ns1-ns0)
    q_cur = Slerp([ns0,ns1],R01)(ns_cur).as_quat()
    t_q_cur = np.hstack([t_cur,q_cur[np.ix_([3,0,1,2])]])  # convert back to qw qx qy qz
    return getT(t_q_cur)

def getPoses(data_dir,save_dir):
    print ('Getting Pose...')

    # image name/time_ns
    ns = np.array(list(np.loadtxt(open(data_dir+'cam0/times.txt'),delimiter=" ")))
    ns = ns[:,0].astype(np.int)

    # pose ground truth
    T_imu_cam0 = np.load(save_dir+"cam0/T_imu_cam0.npy")
    gt_reader = csv.reader(open(data_dir+'gt_imu.csv'),delimiter=",")
    next(gt_reader)
    gt_ns_t_q = np.array(list(gt_reader))
    gt_ns = gt_ns_t_q[:,0].astype(np.int)
    gt_t_q = gt_ns_t_q[:,1:8].astype(np.float32)
    gt_i = 0
    cam1 = np.load(save_dir+"cam1/camera.npy")
    T_0_1 = np.identity(4)
    T_0_1[0,3] = -cam1[4]

    T_w_cam0 = np.empty((len(ns),4,4))
    T_w_cam1 = np.empty((len(ns),4,4))
    for i in tqdm(range(len(ns))):
        # Interpolate pose
        while gt_i<len(gt_ns) and gt_ns[gt_i]<ns[i]:
            gt_i += 1
        
        if gt_i==0:
            T_w_imu = getT(gt_t_q[0])
        elif gt_i==len(gt_ns):
            T_w_imu = getT(gt_t_q[-1])
        else:
            T_w_imu = getInterpolatedT(gt_t_q[gt_i-1],gt_t_q[gt_i],gt_ns[gt_i-1],gt_ns[gt_i],ns[i])
        
        # transform to cam0 frame
        T_w_cam0[i] = np.matmul(T_w_imu,T_imu_cam0)
        T_w_cam1[i] = np.matmul(T_w_cam0[i],T_0_1)

    pose0_path = os.path.join(save_dir,"cam0/poses.npy")
    np.save(pose0_path,T_w_cam0)
    pose1_path = os.path.join(save_dir,"cam1/poses.npy")
    np.save(pose1_path,T_w_cam1)
######################################### get pose for each image #################################################

####################################### get depth using depth filter ##############################################
def calculateCurDepth(cam,px_error_angle,prv_depth,imgs,cur_i,nearby_frames,T_pose):
    h,w = prv_depth.shape

    cur_depth = np.empty_like(prv_depth)
    cur_depth[:] = np.nan    

    # Project depth 
    depth_min = np.inf
    if cur_i>0:
        flow_cur2prv = getFlowEp(imgs[cur_i],imgs[cur_i-1],cam,T_pose[cur_i,cur_i-1],cam,'Consecutive frames GS')
        PROJ_FLOW_DIFF_THRES = 2        # Threshold to accept depth
        for cv in range(h):
            for cu in range(w):
                [fu,fv] = flow_cur2prv[cv,cu,:]
                if np.isnan(fu): continue
                pu,pv = cu+fu,cv+fv
                [cu_p,cv_p,cd] = transferPoint(pu,pv,prv_depth,T_pose[cur_i,cur_i-1],cam,cam)
                if np.isnan(cu_p): continue
                [du,dv] = [cu_p-cu,cv_p-cv]
                if (du*du+dv*dv)<PROJ_FLOW_DIFF_THRES:
                    cur_depth[cv,cu] = cd
                    if depth_min>cd: depth_min = cd
    if depth_min==np.inf:
        depth_min = 1.0
    
    # Initialize seed for pixels without depth
    seeds = []
    z_range = 1.0/depth_min
    a,b = 10,10
    mu,sigma2 = np.nan,np.nan
    vc = 0 # view count
    for v in range(h):
        for u in range(w):
            if np.isnan(cur_depth[v,u]):
                seeds.append(((u,v),a,b,mu,z_range,sigma2,vc))

    # Update seeds using nearby_frames
    cam = (cam[0],cam[1],cam[2],cam[3],px_error_angle)
    for fi in nearby_frames:
        T_cur_fi = T_pose[cur_i,fi]
        flow_cur2fi = getFlowEp(imgs[cur_i],imgs[fi],cam,T_cur_fi,cam,'Depth Filter Flow')
        seeds = filterDepthByFlow(cam,seeds,flow_cur2fi,T_cur_fi)

    # update depth
    SEED_CONVERGE_SIGMA2_THRESH = 200.0
    for seed in seeds:
        [[u,v],a,b,mu,z_range,sigma2,vc] = seed
        if vc>1 and sigma2*sigma2<z_range/SEED_CONVERGE_SIGMA2_THRESH:   
            cur_depth[v,u] = 1.0 / mu
    # print(np.sum(~np.isnan(cur_depth))/h/w)
    return cur_depth  

def getDepth(save_dir):
    print ('Getting Depth...')

    depth_dir = save_dir+"cam0/depth/"
    if os.path.exists(depth_dir): shutil.rmtree(depth_dir)
    os.makedirs(depth_dir)

    # Load images
    img_dir = save_dir+"cam0/images/"
    img_count = len(os.listdir(img_dir))
    imgs = []
    for i in range(img_count):
        img = cv2.imread('{}{}.png'.format(img_dir,i))
        imgs.append(img)
    h,w = imgs[0].shape[:2]

    # Load poses
    T_w = np.load(save_dir+"cam0/poses.npy")
    T_w_inv = np.empty_like(T_w)
    for i in range(img_count):
        T_w_inv[i] = LA.inv(T_w[i])
    T_pose = np.empty((img_count,img_count,4,4))
    for i in range(img_count):
        for j in range(img_count):
            T_i_w = T_w_inv[i]
            T_w_j = T_w[j]
            T_pose[i,j] = np.matmul(T_i_w,T_w_j)

    # Depth filter parameters
    df_dist = 0.1 
    df_angle = 3/180*np.pi
    df_frames = 5
    px_noise = 1.0
    cam = np.load(save_dir+"cam0/camera.npy")
    px_error_angle = np.arctan(px_noise/(2.0*np.fabs(cam[0])))*2.0 # px_error_angle

    # Get depth by projection and depth filtering
    prv_depth = np.empty((h,w))  
    prv_depth[:] = np.nan 
    for cur_i in tqdm(range(img_count)):
        # Find good nearby frames to calculate depth
        nearby_frames = []
        s=e=cur_i
        while e<img_count and LA.norm(T_pose[e,s][0:3,3])<df_dist: e+=1
        while len(nearby_frames)<df_frames:
            while e<img_count and LA.norm(T_pose[e,s][0:3,3])<df_dist \
            and LA.norm(Rotation.from_matrix(T_pose[e,s][0:3,0:3]).as_rotvec())<df_angle:
                e+=1
            if e>=img_count:
                break
            nearby_frames.append(e)
            s=e

        # Current depth
        cur_depth = calculateCurDepth(cam,px_error_angle,prv_depth,imgs,cur_i,nearby_frames,T_pose)
        fname = os.path.join(depth_dir,str(cur_i))
        np.save(fname,cur_depth)
        
        prv_depth = cur_depth

####################################### get depth using depth filter ##############################################

###################################### get unrolling ground-truth flow #############################################
# @jit
def getGSRSFlow(flow10,depth0,cam0,cam1,T_1_0):
    h,w = flow10.shape[:2]
    flow_gs2rs = np.empty_like(flow10)
    flow_gs2rs[:] = np.nan
    flow_rs2gs = np.empty_like(flow10)
    flow_rs2gs[:] = np.nan

    # Project from cam0 to cam1 
    for v1 in range(h):
        for u1 in range(w): 
            uf,vf = flow10[v1,u1,:]
            if np.isnan(uf): continue
            u0,v0 = u1+uf,v1+vf
            [u1gs,v1gs,_] = transferPoint(u0,v0,depth0,T_1_0,cam0,cam1)
            if np.isnan(u1gs): continue
            [u1gsi,v1gsi] = [int(u1gs+0.5),int(v1gs+0.5)]
            if 0<=u1gsi<w and 0<=v1gsi<h:
                flow_gs2rs[v1gsi,u1gsi,0] = u1-u1gs
                flow_gs2rs[v1gsi,u1gsi,1] = v1-v1gs
                flow_rs2gs[v1,u1,0] = u1gs-u1
                flow_rs2gs[v1,u1,1] = v1gs-v1
                            
    return [flow_gs2rs,flow_rs2gs]

def getGSRSFlows(save_dir):
    print ('Getting Unrolling Flow...')

    img0_dir = save_dir+"cam0/images/"
    depth0_dir = save_dir+"cam0/depth/"
    img1_dir = save_dir+"cam1/images/"
    flows_gs2rs_dir = save_dir+"cam1/flows_gs2rs/"
    if os.path.exists(flows_gs2rs_dir): shutil.rmtree(flows_gs2rs_dir)
    os.makedirs(flows_gs2rs_dir)
    flows_rs2gs_dir = save_dir+"cam1/flows_rs2gs/"
    if os.path.exists(flows_rs2gs_dir): shutil.rmtree(flows_rs2gs_dir)
    os.makedirs(flows_rs2gs_dir)

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
        [flow_gs2rs,flow_rs2gs] = getGSRSFlow(flow10,depth0,cam0,cam1,T_1_0)
        gs2rs_name = os.path.join(flows_gs2rs_dir,str(i))
        np.save(gs2rs_name,flow_gs2rs)
        rs2gs_name = os.path.join(flows_rs2gs_dir,str(i))
        np.save(rs2gs_name,flow_rs2gs)

        # h,w = img1.shape[:2]
        # img1_gs = np.zeros_like(img1)
        # indy, indx = np.indices((h,w), dtype=np.float32)
        # map_x = indx.reshape(h,w).astype(np.float32) + flow_gs2rs[:,:,0]
        # map_y = indy.reshape(h,w).astype(np.float32) + flow_gs2rs[:,:,1]
        # img1_gs = cv2.remap(img1, map_x, map_y, cv2.INTER_LINEAR)
        # cv2.namedWindow('img1_gs', flags=cv2.WINDOW_NORMAL)
        # cv2.imshow('img1_gs',img1_gs)
        # cv2.waitKey(1)  

###################################### get unrolling ground-truth flow #############################################

def getNextRSFlows(save_dir):
    print ('Getting Next RS Flow...')

    img1_dir = save_dir+"cam1/images/"
    flows_next_rs_dir = save_dir+"cam1/flows_next_rs/"
    if os.path.exists(flows_next_rs_dir): shutil.rmtree(flows_next_rs_dir)
    os.makedirs(flows_next_rs_dir)

    img_count = len(os.listdir(img1_dir))
    for i in tqdm(range(img_count-1)):
        img_cur = cv2.imread('{}{}.png'.format(img1_dir,i))
        img_nxt = cv2.imread('{}{}.png'.format(img1_dir,i+1))
        flow_next_rs = getFlowBD(img_cur,img_nxt,'Consecutive frames RS')
        next_rs_name = os.path.join(flows_next_rs_dir,str(i))
        np.save(next_rs_name,flow_next_rs)

def getNextEpipolarLines(save_dir):
    print ('Getting Epipolar Lines...')

    epipolar_lines_dir = save_dir+"cam1/epipolar_lines/"
    if os.path.exists(epipolar_lines_dir): shutil.rmtree(epipolar_lines_dir)
    os.makedirs(epipolar_lines_dir)

    cam = np.load(save_dir+"cam1/camera.npy")
    img1_dir = save_dir+"cam1/images/"
    img_count = len(os.listdir(img1_dir))
    img10 = cv2.imread('{}0.png'.format(img1_dir))
    h,w = img10.shape[:2]

    # Load poses
    T_w = np.load(save_dir+"cam1/poses.npy")

    for i in tqdm(range(img_count-1)):
        epi_mat = np.empty((h,w,3))
        epi_mat[:] = np.nan

        T_c_w = LA.inv(T_w[i])
        T_w_n = T_w[i+1]
        T_cur_nxt = np.matmul(T_c_w,T_w_n)

        [EH, degenerated] = getEH(cam,T_cur_nxt,cam)
        for cv in range(h):
            for cu in range(w): 
                [a,b,c] = getEpipolarLine(cu,cv,EH,degenerated)
                if not np.isnan(a): 
                    epi_mat[cv,cu,0] = a
                    epi_mat[cv,cu,1] = b
                    epi_mat[cv,cu,2] = c

        fname = os.path.join(epipolar_lines_dir,str(i))
        np.save(fname,epi_mat)
        

if __name__=="__main__":
    resolution = (320,256)
    seqs = ['1','3','5','2','4','7','9','6','8','10']
    for seq in seqs:
        print ('\n\n\nProcessing Sequence '+str(seq))
        
        data_dir = '/home/moxxx066/Workspace/data/dataset-seq{}/dso/'.format(seq)
        save_dir = os.path.join(os.getcwd(),'../data/seq{}/'.format(seq))

        # if os.path.exists(save_dir): shutil.rmtree(save_dir)
        # os.makedirs(save_dir)
        # stereoRectify(data_dir,save_dir,resolution)
        getPoses(data_dir,save_dir)
        # getDepth(save_dir)

        getGSRSFlows(save_dir)
        getNextRSFlows(save_dir)
        getNextEpipolarLines(save_dir)