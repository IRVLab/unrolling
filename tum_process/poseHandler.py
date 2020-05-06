from __future__ import absolute_import,division,print_function
from scipy.spatial.transform import Rotation, RotationSpline
from scipy import interpolate
import numpy as np
import os
import csv
from tqdm import tqdm
from numpy import linalg as LA

class Pose:
    def __init__(self,gt_file,T_gt_target):
        gt_reader = csv.reader(open(gt_file),delimiter=",")
        next(gt_reader) # skip the header line
        gt_ns_t_q = np.array(list(gt_reader))
        gt_ns = np.array(gt_ns_t_q[:,0]).astype(np.long)
        gt_t_q = np.array(gt_ns_t_q[:,1:]).astype(np.float)
        self.first_ns = gt_ns[1]
        self.last_ns = gt_ns[-2]

        # transfer to target coordinate
        ts,Rs = [],[]
        for i in range(len(gt_ns)):
            T_w_gt = np.identity(4)
            T_w_gt[0:3,3] = gt_t_q[i,:3]
            gt_q = gt_t_q[i,np.ix_([4,5,6,3])]    # convert to qx qy qz qw for from_quat
            T_w_gt[0:3,0:3] = Rotation.from_quat(gt_q).as_matrix()
            T_w_target = np.matmul(T_w_gt,T_gt_target)
            ts.append(T_w_target[0:3,3])
            Rs.append(T_w_target[0:3,0:3])

        # splines
        self.t_spline = interpolate.CubicSpline(gt_ns, ts)
        self.R_spline = RotationSpline(gt_ns, Rotation.from_matrix(Rs))

    def isValidNs(self, query_ns):
        return self.first_ns<query_ns<self.last_ns

    def getPoseAt(self, query_ns):
        T = np.identity(4)
        T[0:3,0:3] = self.R_spline(query_ns).as_matrix()
        T[0:3,3] = self.t_spline(query_ns)
        return T

    def getVelBetween(self, ns_a, ns_b):
        dur_s = (ns_b-ns_a) / 1e9
        T_w_0 = self.getPoseAt(ns_a)
        T_w_1 = self.getPoseAt(ns_b)
        T_0_1 = np.matmul(LA.inv(T_w_0),T_w_1)
        vt = T_0_1[0:3,3] / dur_s
        vr = Rotation.from_matrix(T_0_1[0:3,0:3]).as_rotvec() / dur_s
        return np.array([vt[0],vt[1],vt[2],vr[0],vr[1],vr[2]])

    def getAccBetween(self, ns_a, ns_b):
        dur_s = (ns_b-ns_a) / 1e9 / 2
        v0 = self.getVelBetween(ns_a, (ns_a+ns_b)/2)
        v1 = self.getVelBetween((ns_a+ns_b)/2, ns_b)
        v0[3:] = Rotation.from_rotvec(v0[3:]).as_euler('zxy')
        v1[3:] = Rotation.from_rotvec(v1[3:]).as_euler('zxy')
        atar = (v1-v0) / dur_s
        return atar

def getPoses(data_dir,save_dir,img_h,ns_per_v):
    # image name/time_ns
    img_ns = []
    times = open(data_dir+'cam0/times.txt').read().splitlines()
    for i in range(1,len(times)):
        cur_ns = times[i].split(' ')[0]
        img_ns.append(int(cur_ns))

    # pose ground truth
    cam1 = np.load(save_dir+"cam1/camera.npy")
    T_cam0_cam1 = np.identity(4)
    T_cam0_cam1[0,3] = -cam1[4]
    T_imu_cam0 = np.load(save_dir+"cam0/T_imu_cam0.npy")
    T_imu_cam1 = np.matmul(T_imu_cam0,T_cam0_cam1)
    gt_pose_cam1 = Pose(data_dir+'gt_imu.csv', T_imu_cam1)

    valid_ns,T_cam0_v1,vel_t_r,acc_t_r = [],[],[],[]
    for i in tqdm(range(len(img_ns))):
        if not gt_pose_cam1.isValidNs(img_ns[i]+ns_per_v*(img_h-1)) \
            or not gt_pose_cam1.isValidNs(img_ns[i]):
            # print([gt_pose_cam1.first_ns,img_ns[i],gt_pose_cam1.last_ns])
            continue
        valid_ns.append(img_ns[i])

        T_w_cam1 = gt_pose_cam1.getPoseAt(img_ns[i])
        T_cam0_w = np.matmul(T_cam0_cam1, LA.inv(T_w_cam1))

        # get pose for each scan line
        T_cam0_v1_i =[]
        for v in range(img_h):
            T_w_v1 = gt_pose_cam1.getPoseAt(img_ns[i]+ns_per_v*v)
            T_cam0_v1_i.append(np.matmul(T_cam0_w,T_w_v1))
        T_cam0_v1.append(T_cam0_v1_i)

        # get velocity and acceleration data for comparision
        vel_t_r.append(gt_pose_cam1.getVelBetween(img_ns[i],img_ns[i]+ns_per_v*(img_h-1)))
        acc_t_r.append(gt_pose_cam1.getAccBetween(img_ns[i],img_ns[i]+ns_per_v*(img_h-1)))

    ns_path = os.path.join(save_dir,"valid_ns.npy")
    np.save(ns_path,np.array(valid_ns))

    pose1_path = os.path.join(save_dir,"poses_cam0_v1.npy")
    np.save(pose1_path,np.array(T_cam0_v1))

    vel_path = os.path.join(save_dir,"cam1/vel_t_r.npy")
    np.save(vel_path,np.array(vel_t_r))

    acc_path = os.path.join(save_dir,"cam1/acc_t_r.npy")
    np.save(acc_path,np.array(acc_t_r))