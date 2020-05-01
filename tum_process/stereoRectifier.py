from __future__ import absolute_import,division,print_function
import numpy as np
import cv2
import os
import shutil
import yaml
from numpy import linalg as LA
from tqdm import tqdm

def saveCamera(fileName,P):
    fxfycxcytx = np.array([P[0,0],P[1,1],P[0,2],P[1,2],P[0,3]/P[0,0]])
    np.save(fileName,fxfycxcytx)

def stereoRectify(data_dir,save_dir,resolution):
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