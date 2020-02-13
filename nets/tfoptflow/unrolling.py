"""
pwcnet_predict_from_img_pairs.py

Run inference on a list of images pairs.

Written by Phil Ferriere

Licensed under the MIT License (see LICENSE for details)
"""

from __future__ import absolute_import, division, print_function
from copy import deepcopy
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import display_img_pairs_w_flows

import os
import numpy as np
from numpy import linalg as LA
import cv2
import csv
from matplotlib import pyplot as plt
from sklearn import linear_model

img0_path = '/home/jiawei/Workspace/unrolling/data/tum/seq1/img0/'
img1_path = '/home/jiawei/Workspace/unrolling/data/tum/seq1/img1/'

# TODO: Set device to use for inference
# Here, we're using a GPU (use '/device:CPU:0' to run inference on the CPU)
gpu_devices = ['/device:GPU:0']  
controller = '/device:GPU:0'

# TODO: Set the path to the trained model (make sure you've downloaded it first from http://bit.ly/tfoptflow)
ckpt_path = '/home/jiawei/Workspace/unrolling/data/sintel_gray_weights/pwcnet.sintel_gray.ckpt-54000'

# Configure the model for inference, starting with the default options
nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
nn_opts['verbose'] = False
nn_opts['ckpt_path'] = ckpt_path
nn_opts['batch_size'] = 1
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller

# We're running the PWC-Net-large model in quarter-resolution mode
# That is, with a 6 level pyramid, and upsampling of level 2 by 4 in each dimension as the final flow prediction
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2
nn_opts['resize'] = False

# The size of the images in this dataset are not multiples of 64, while the model generates flows padded to multiples
# of 64. Hence, we need to crop the predicted flows to their original size
# nn_opts['adapt_info'] = (1, 436, 1024, 2)

nn = ModelPWCNet(mode='test', options=nn_opts)

def draw_flow(img, flow, mask, step=2):
    h, w = img.shape[:2]
    vis = img
    for di in range(0,h,step):
        for dj in range(0,w,step):
            if mask[di, dj]: 
                xf, yf = flow[di,dj].T
                cv2.line(vis, (dj, di), (np.int32(dj+xf+0.5), np.int32(di+yf+0.5)), (0, 255, 0))

    return vis

def cal_err_mask(flow, mask):
    h, w = flow.shape[:2]

    err = 0.0
    count = 0
    for di in range(0,h,10):
        for dj in range(0,w,10):
            if mask[di, dj]: 
                xf, yf = flow[di,dj].T
                err = err + yf*yf
                count = count + 1
    return err / count


def cal_flow_err(img_pairs):
    # Estimate optical flow using PWC-New
    flows = nn.predict_from_img_pairs(img_pairs, batch_size=1, verbose=False)

    # Calculate errors
    errs = []
    for i in range(len(img_pairs)):
        edges = cv2.Canny(img_pairs[i][0],20,50)
        errs_new = cal_err_mask(flows[i], edges)
        print(errs_new)
        errs.append(errs_new)
        cv2.imshow('flow', draw_flow(img_pairs[i][0], flows[i], edges))
        cv2.waitKey(1)
    
    return errs

def main():
    # Build a list of image pairs 
    img0_filenames = sorted(os.listdir(img0_path))
    img1_filenames = sorted(os.listdir(img1_path))
    img_pairs = []
    errs = []
    for i in range(len(img0_filenames)):
    # for i in range(100,200):
        img_path1 = os.path.join(img0_path, img0_filenames[i])
        img_path2 = os.path.join(img1_path, img1_filenames[i])
        img1, img2 = cv2.imread(img_path1), cv2.imread(img_path2)
        img_pairs.append((img1, img2))
        if len(img_pairs)>10:
            err_new = cal_flow_err(img_pairs)
            errs.append(err_new)
            img_pairs = []

    err_new = cal_flow_err(img_pairs)
    errs.append(err_new)
    print(f'\n\n{errs}')
    # np.savetxt('errs.txt', np.asarray(errs).ravel())

if __name__ == '__main__':
    main()