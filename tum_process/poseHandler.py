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
        next(gt_reader)
        gt_ns_t_q = np.array(list(gt_reader)).astype(np.float)
        gt_ns = gt_ns_t_q[:,0]

        # transfer to target coordinate
        ts,Rs = [],[]
        for i in range(len(gt_ns)):
            gt_q = gt_ns_t_q[i,np.ix_([5,6,7,4])].astype(np.float)    # convert to qx qy qz qw for from_quat
            T_w_gt = np.identity(4)
            T_w_gt[0:3,0:3] = Rotation.from_quat(gt_q).as_matrix()
            T_w_gt[0:3,3] = gt_ns_t_q[i,1:4].astype(np.float)
            T_w_target = np.matmul(T_w_gt,T_gt_target)
            ts.append(T_w_target[0:3,3])
            Rs.append(T_w_target[0:3,0:3])

        # splines
        self.t_spline = interpolate.CubicSpline(gt_ns, ts)
        self.R_spline = RotationSpline(gt_ns, Rotation.from_matrix(Rs))

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
        vtvr = [vt[0],vt[1],vt[2],vr[0],vr[1],vr[2]]
        return vtvr

    def getAccBetween(self, ns_a, ns_b):
        dur_s = (ns_b-ns_a) / 1e9
        v0 = self.getVelBetween(ns_a, (ns_a+ns_b)/2)
        v1 = self.getVelBetween((ns_a+ns_b)/2, ns_b)
        atar = [v1[0]-v0[0],v1[1]-v0[1],v1[2]-v0[2],v1[3]-v0[3],v1[4]-v0[4],v1[5]-v0[5]] / dur_s
        return atar

def getPoses(data_dir,save_dir,img_h,ns_per_v):
    # image name/time_ns
    img_ns = np.array(list(np.loadtxt(open(data_dir+'cam0/times.txt'),delimiter=" ")))
    img_ns = img_ns[:,0].astype(np.int)

    # pose ground truth
    cam1 = np.load(save_dir+"cam1/camera.npy")
    T_cam0_cam1 = np.identity(4)
    T_cam0_cam1[0,3] = -cam1[4]
    T_imu_cam0 = np.load(save_dir+"cam0/T_imu_cam0.npy")
    T_imu_cam1 = np.matmul(T_imu_cam0,T_cam0_cam1)
    gt_pose_cam1 = Pose(data_dir+'gt_imu.csv', T_imu_cam1)

    T_cam0_v1 = np.empty((len(img_ns),img_h,4,4))
    vel_t_r = np.empty((len(img_ns),6))
    acc_t_r = np.empty((len(img_ns),6))
    for i in tqdm(range(len(img_ns))):
        T_w_cam1 = gt_pose_cam1.getPoseAt(img_ns[i])
        T_cam0_w = np.matmul(T_cam0_cam1, LA.inv(T_w_cam1))

        # get pose for each scan line
        for v in range(img_h):
            T_w_v1 = gt_pose_cam1.getPoseAt(img_ns[i]+ns_per_v*v)
            T_cam0_v1[i,v] = np.matmul(T_cam0_w,T_w_v1)

        # get velocity and acceleration data for comparision
        vel_t_r[i,:] = gt_pose_cam1.getVelBetween(img_ns[i],img_ns[i]+ns_per_v*(img_h-1))
        acc_t_r[i,:] = gt_pose_cam1.getAccBetween(img_ns[i],img_ns[i]+ns_per_v*(img_h-1))

    pose1_path = os.path.join(save_dir,"poses_cam0_v1.npy")
    np.save(pose1_path,T_cam0_v1)

    vel_path = os.path.join(save_dir,"cam1/vel_t_r.npy")
    np.save(vel_path,vel_t_r)

    acc_path = os.path.join(save_dir,"cam1/acc_t_r.npy")
    np.save(acc_path,acc_t_r)