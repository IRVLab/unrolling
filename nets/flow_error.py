#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from numpy import linalg as LA
import cv2
from matplotlib import pyplot as plt
from sklearn import linear_model

img0_path = '/mnt/data2/unRolling_Shutter/Rolling_Shutter_TUM/seq1/img0/'
img1_path = '/mnt/data2/unRolling_Shutter/Rolling_Shutter_TUM/seq1/img1/'


def squashFlowError(flow, mask):
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


def getFlowCV(img0, img1):
    return cv2.calcOpticalFlowFarneback(img0, img1, None, 0.5, 1, 25, 3, 7, 1.5, 0)


def calFlowAndError(img0, img1):
    ## Estimate optical flow
    flows = getFlowCV(img0, img1)
    # Calculate errors
    edges = cv2.Canny(img1, 20, 50)
    return squashFlowError(flows, edges)


def main():
    # Build a list of image pairs 
    img0_filenames = sorted(os.listdir(img0_path))
    img1_filenames = sorted(os.listdir(img1_path))
    img_pairs = []
    errs = []
    for i in range(len(img0_filenames)):
    # for i in range(100,200):
        img_path0 = os.path.join(img0_path, img0_filenames[i])
        img_path1 = os.path.join(img1_path, img1_filenames[i])
        img0, img1 = cv2.imread(img_path0, 0), cv2.imread(img_path1, 0)
        print (calFlowAndError(img0, img1))

        print ((img0.shape, img1.shape))

if __name__ == '__main__':
    main()



