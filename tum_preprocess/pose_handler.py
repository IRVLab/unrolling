from __future__ import absolute_import,division,print_function
from scipy.spatial.transform import Rotation, RotationSpline
from scipy import interpolate
import numpy as np
import os
import csv
from tqdm import tqdm
from numpy import linalg as LA

class Pose:
    def __init__(self,gt_file):
        gt_reader = csv.reader(open(gt_file),delimiter=",")
        next(gt_reader)
        gt_ns_t_q = np.array(list(gt_reader))
        gt_ns = gt_ns_t_q[:,0].astype(np.int)
        # translation
        self.tx_spline = interpolate.splrep(gt_ns, gt_ns_t_q[:,1].astype(np.float), s=0)
        self.ty_spline = interpolate.splrep(gt_ns, gt_ns_t_q[:,2].astype(np.float), s=0)
        self.tz_spline = interpolate.splrep(gt_ns, gt_ns_t_q[:,3].astype(np.float), s=0)
        # rotation
        gt_q = gt_ns_t_q[:,np.ix_([5,6,7,4])].astype(np.float)    # convert to qx qy qz qw for from_quat
        gt_q = gt_q[:,0,:]
        gt_R = Rotation.from_quat(gt_q)
        self.R_spline = RotationSpline(gt_ns, gt_R)

    def getPoseAt(self, query_ns):
        T = np.identity(4)
        T[0:3,0:3] = self.R_spline(query_ns).as_matrix()
        T[0,3] = interpolate.splev(query_ns, self.tx_spline)
        T[1,3] = interpolate.splev(query_ns, self.ty_spline)
        T[2,3] = interpolate.splev(query_ns, self.tz_spline)
        return T

def getPoses(data_dir,save_dir,resolution,ns_per_v):
    # image name/time_ns
    img_ns = np.array(list(np.loadtxt(open(data_dir+'cam0/times.txt'),delimiter=" ")))
    img_ns = img_ns[:,0].astype(np.int)

    # pose ground truth
    gt_pose = Pose(data_dir+'gt_imu.csv')
    T_imu_cam0 = np.load(save_dir+"cam0/T_imu_cam0.npy")
    cam1 = np.load(save_dir+"cam1/camera.npy")
    T_0_1 = np.identity(4)
    T_0_1[0,3] = -cam1[4]
    T_imu_cam1 = np.matmul(T_imu_cam0,T_0_1)

    T_w_cam0 = np.empty((len(img_ns),4,4))
    T_cam0_v1 = np.empty((len(img_ns),resolution[1],4,4))
    for i in tqdm(range(len(img_ns))):
        # transform to cam0 frame
        T_w_imu = gt_pose.getPoseAt(img_ns[i])
        T_w_cam0[i] = np.matmul(T_w_imu,T_imu_cam0)
        T_cam0_w = LA.inv(T_w_cam0[i])

        for v in range(resolution[1]):
            T_w_imu = gt_pose.getPoseAt(img_ns[i]+ns_per_v*v)
            T_w_v1 = np.matmul(T_w_imu,T_imu_cam1)
            T_cam0_v1[i,v] = np.matmul(T_cam0_w,T_w_v1)

    pose0_path = os.path.join(save_dir,"poses_cam0.npy")
    np.save(pose0_path,T_w_cam0)
    pose1_path = os.path.join(save_dir,"poses_v1.npy")
    np.save(pose1_path,T_cam0_v1)