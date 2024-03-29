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
import numpy as np
import cv2
import os
import yaml
from numpy import linalg as LA
from tqdm import tqdm


def stereoRectify(data_path, save_path, resolution):
    img0_path = save_path+"cam0/images/"
    img1_path = save_path+"cam1/images/"
    if not os.path.exists(img0_path):
        os.makedirs(img0_path)
    if not os.path.exists(img1_path):
        os.makedirs(img1_path)

    # Read original calibration file
    with open(data_path+"camchain.yaml") as file:
        camchain = yaml.load(file, Loader=yaml.FullLoader)

        imageSize = tuple(camchain['cam0']['resolution'])
        cam0_intrinsics = camchain['cam0']['intrinsics']
        K0 = np.matrix([[cam0_intrinsics[0], 0, cam0_intrinsics[2]],
                        [0, cam0_intrinsics[1], cam0_intrinsics[3]],
                        [0, 0, 1]])
        D0 = np.array(camchain['cam0']['distortion_coeffs'])

        cam1_intrinsics = camchain['cam1']['intrinsics']
        K1 = np.matrix([[cam1_intrinsics[0], 0, cam1_intrinsics[2]],
                        [0, cam1_intrinsics[1], cam1_intrinsics[3]],
                        [0, 0, 1]])
        D1 = np.array(camchain['cam1']['distortion_coeffs'])

        T01 = np.matrix(camchain['cam1']['T_cn_cnm1'])
        R = T01[np.ix_([0, 1, 2], [0, 1, 2])]
        tvec = T01[np.ix_([0, 1, 2], [3])]

    # Fisheye stere0 rectify
    R0, R1, P0, P1, Q = cv2.fisheye.stereoRectify(
        K0, D0, K1, D1, imageSize, R, tvec, 0, newImageSize=resolution)
    map0 = cv2.fisheye.initUndistortRectifyMap(
        K0, D0, R0, P0, resolution, cv2.CV_32F)
    map1 = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, R1, P1, resolution, cv2.CV_32F)
    np.save(save_path+"cam0/stereo_map.npy", np.array(map0))
    np.save(save_path+"cam1/stereo_map.npy", np.array(map1))

    # Loopup table for rolling shutter time query
    cols_idx, _ = np.indices((imageSize[1], imageSize[0]), dtype=np.float32)
    cols_idx = cols_idx / imageSize[1] * resolution[1]
    v1_lut = cv2.remap(cols_idx, map1[0], map1[1], cv2.INTER_NEAREST)
    v1_lut = np.array(v1_lut, dtype=int)
    np.save(save_path+"cam1/v1_lut.npy", v1_lut)

    fxfycxcytx0 = np.array(
        [P0[0, 0], P0[1, 1], P0[0, 2], P0[1, 2], P0[0, 3]/P0[0, 0]])
    fxfycxcytx1 = np.array(
        [P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2], P1[0, 3]/P1[0, 0]])
    np.save(save_path+"cam0/camera.npy", fxfycxcytx0)
    np.save(save_path+"cam1/camera.npy", fxfycxcytx1)

    T_cam0_imu = np.matrix(camchain['cam0']['T_cam_imu'])
    T2rectified = np.identity(4)
    T2rectified[0:3, 0:3] = R0
    T_imu_cam0 = LA.inv(T2rectified*T_cam0_imu)
    np.save(save_path+"cam0/T_imu_cam0.npy", T_imu_cam0)


def stereoRemap(data_path, save_path):
    valid_ns = np.load(save_path+"valid_ns.npy")
    map0 = np.load(save_path+"cam0/stereo_map.npy")
    map1 = np.load(save_path+"cam1/stereo_map.npy")
    maps = [map0, map1]
    # Remap images
    for i in tqdm(range(valid_ns.shape[0])):
        for cam_i in [0, 1]:
            img = cv2.imread(
                '{}cam{}/images/{}.png'.format(data_path, cam_i, valid_ns[i]))
            img_rect = cv2.remap(
                img, maps[cam_i][0], maps[cam_i][1], cv2.INTER_LINEAR)
            save_file = '{}cam{}/images/{}.png'.format(save_path, cam_i, i)
            cv2.imwrite(save_file, img_rect)
