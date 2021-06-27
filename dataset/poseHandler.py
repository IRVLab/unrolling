# Learning Rolling Shutter Correction from Real Data without Camera Motion Assumption
# Copyright (C) <2021> <Jiawei Mo, Md Jahidul Islam, Junaed Sattar>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import absolute_import, division, print_function
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.interpolate import interp1d, CubicSpline
import numpy as np
import os
import csv
from tqdm import tqdm
from numpy import linalg as LA


class Imu:
    def __init__(self, imu_file, R_cam1_imu):
        imu_reader = csv.reader(open(imu_file), delimiter=" ")
        next(imu_reader)  # skip the header line
        imu_ns_w_a = np.array(list(imu_reader))
        imu_ns = np.array(imu_ns_w_a[:, 0], dtype=np.long)
        imu_w_a = np.array(imu_ns_w_a[:, 1:], dtype=np.float)
        self.first_ns = imu_ns[1]
        self.last_ns = imu_ns[-2]

        cam1_w_a = np.zeros((len(imu_ns), 6))
        for i in range(len(imu_ns)):
            cam1_w_a[i, :3] = np.matmul(R_cam1_imu, imu_w_a[i, :3])
            cam1_w_a[i, 3:] = np.matmul(R_cam1_imu, imu_w_a[i, 3:])

        self.imu_interp = interp1d(imu_ns, cam1_w_a, axis=0)

    def isValidNs(self, query_ns):
        return self.first_ns <= query_ns <= self.last_ns

    def getImuAt(self, query_ns):
        return self.imu_interp(query_ns)


class Pose:
    def __init__(self, gt_file, T_imu_cam1):
        gt_reader = csv.reader(open(gt_file), delimiter=",")
        next(gt_reader)  # skip the header line
        gt_ns_t_q = np.array(list(gt_reader))
        gt_ns = np.array(gt_ns_t_q[:, 0], dtype=np.long)
        gt_t_q = np.array(gt_ns_t_q[:, 1:], dtype=np.float)
        self.first_ns = gt_ns[1]
        self.last_ns = gt_ns[-2]

        # transfer to target coordinate
        ts, Rs = [], []
        for i in range(len(gt_ns)):
            T_w_gt = np.identity(4)
            T_w_gt[0:3, 3] = gt_t_q[i, :3]
            # convert to qx qy qz qw for from_quat
            gt_q = gt_t_q[i, np.ix_([4, 5, 6, 3])]
            T_w_gt[0:3, 0:3] = Rotation.from_quat(gt_q).as_matrix()
            T_w_cam1 = np.matmul(T_w_gt, T_imu_cam1)
            ts.append(T_w_cam1[0:3, 3])
            Rs.append(T_w_cam1[0:3, 0:3])

        # splines
        self.t_spline = CubicSpline(gt_ns, ts)
        self.R_spline = RotationSpline(gt_ns, Rotation.from_matrix(Rs))

    def isValidNs(self, query_ns):
        return self.first_ns <= query_ns <= self.last_ns

    def getPoseAt(self, query_ns):
        T = np.identity(4)
        T[0:3, 0:3] = self.R_spline(query_ns).as_matrix()
        T[0:3, 3] = self.t_spline(query_ns)
        return T


def getPoses(data_path, save_path, img_h, ns_per_v):
    # image name/time_ns
    img_ns = []
    times = open(data_path+'cam0/times.txt').read().splitlines()
    for i in range(1, len(times)):
        cur_ns = times[i].split(' ')[0]
        img_ns.append(int(cur_ns))

    # pose ground truth
    cam1 = np.load(save_path+"cam1/camera.npy")
    T_cam0_cam1 = np.identity(4)
    T_cam0_cam1[0, 3] = -cam1[4]
    T_imu_cam0 = np.load(save_path+"cam0/T_imu_cam0.npy")
    T_imu_cam1 = np.matmul(T_imu_cam0, T_cam0_cam1)
    gt_pose_cam1 = Pose(data_path+'gt_imu.csv', T_imu_cam1)
    imu_cam1 = Imu(data_path+'imu.txt', np.transpose(T_imu_cam1[:3, :3]))

    valid_ns, T_cam0_v1, pose_w_cam1, pose_cam1_v1, imu_cam1_v1 = [], [], [], [], []
    for i in tqdm(range(len(img_ns))):
        if not (gt_pose_cam1.isValidNs(img_ns[i]+ns_per_v*(img_h-1))
                and gt_pose_cam1.isValidNs(img_ns[i])
                and imu_cam1.isValidNs(img_ns[i]+ns_per_v*(img_h-1))
                and imu_cam1.isValidNs(img_ns[i])):
            continue
        valid_ns.append(img_ns[i])

        T_w_cam1 = gt_pose_cam1.getPoseAt(img_ns[i])
        t = T_w_cam1[0:3, 3]
        q = Rotation.from_matrix(T_w_cam1[0:3, 0:3]).as_quat()
        pose_w_cam1.append(
            np.array([i, t[0], t[1], t[2], q[0], q[1], q[2], q[3]]))

        T_cam1_w = LA.inv(T_w_cam1)
        T_cam0_w = np.matmul(T_cam0_cam1, T_cam1_w)
        # get pose for each scan line
        T_cam0_v1_i = []
        pose_cam1_v1_i = np.zeros((img_h, 6))
        imu_cam1_v1_i = np.zeros((img_h, 6))
        for v in range(img_h):
            # row-wise imu
            imu_cam1_v1_i[v] = imu_cam1.getImuAt(img_ns[i]+ns_per_v*v)
            # row-wise pose
            T_w_v1 = gt_pose_cam1.getPoseAt(img_ns[i]+ns_per_v*v)
            T_cam0_v1_i.append(np.matmul(T_cam0_w, T_w_v1))

            T_cam1_v1_i = np.matmul(T_cam1_w, T_w_v1)
            t = T_cam1_v1_i[0:3, 3]
            r = Rotation.from_matrix(T_cam1_v1_i[0:3, 0:3]).as_rotvec()
            pose_cam1_v1_i[v] = [t[0], t[1], t[2], r[0], r[1], r[2]]
        T_cam0_v1.append(T_cam0_v1_i)
        pose_cam1_v1.append(pose_cam1_v1_i)
        imu_cam1_v1.append(imu_cam1_v1_i)

    ns_path = os.path.join(save_path, "valid_ns.npy")
    np.save(ns_path, np.array(valid_ns))

    pose0_path = os.path.join(save_path, "T_cam0_v1.npy")
    np.save(pose0_path, np.array(T_cam0_v1))

    pose1w_path = os.path.join(save_path, "cam1/pose_w_cam1.txt")
    np.savetxt(pose1w_path, np.array(pose_w_cam1), delimiter=',')

    posev1_path = os.path.join(save_path, "cam1/pose_cam1_v1.npy")
    np.save(posev1_path, np.array(pose_cam1_v1))

    imu_cam1_path = os.path.join(save_path, "cam1/imu_cam1_v1.npy")
    np.save(imu_cam1_path, np.array(imu_cam1_v1))
