from __future__ import print_function, division
import os
import cv2
import numpy as np
# keras libs
from keras.optimizers import Adam
# local libs
from utils import dataLoaderTUM, draw_flow
from hfnet import Res_HFNet

def rectify_img_by_flow_batch(imgs, flows):
    b, h, w = imgs.shape[:3]
    rectified_imgs = np.zeros((b,h,w,1), dtype=np.uint8)
    indy, indx = np.indices((h, w), dtype=np.float32)
    map_x = indx.reshape(h, w).astype(np.float32)
    map_y = indy.reshape(h, w).astype(np.float32)
    for bi in range(b):
        map_x_bi = map_x - flows[bi,:,:,0]
        map_y_bi = map_y - flows[bi,:,:,1]
        rectified_img = cv2.remap(cv2.cvtColor(imgs[bi], cv2.COLOR_GRAY2RGB), map_x_bi, map_y_bi, cv2.INTER_LINEAR)
        rectified_imgs[bi,:,:,0] = cv2.cvtColor(rectified_img, cv2.COLOR_RGB2GRAY)

    return rectified_imgs

## dataset and experiment directories
dataset_name = "TUM"
# data_dir = "/mnt/data2/unRolling_Shutter/Rolling_Shutter_TUM/"
data_dir = "/home/jiawei/Workspace/data/datasets/TUM/"
seq = 1 # None for all

## parameters
im_shape = (256, 320) # should be multiples of 64 to avoid PWC padding
data_loader = dataLoaderTUM(data_path=data_dir, seq_no=seq, out_res=(im_shape[1], im_shape[0]), load_flow=True) 

model_loader = Res_HFNet(im_shape)
model_loader.model.load_weights('/home/jiawei/Workspace/unrolling/checkpoints/TUM/model.h5')
# model_loader.model.compile(optimizer=Adam(1e-4, 0.5), loss='mse')

## training pipeline
print (f"\Testing: HFNet with {dataset_name} data")
wins = 0
for i in range(data_loader.num_train):
    imgs_gs, imgs_rs, flows_rs2gen, _  = data_loader.load_batch(i, batch_size=1)

    flow_rs2gen_pred = model_loader.model.predict(imgs_rs)
    prv_diff = np.mean(np.square(flows_rs2gen))
    new_diff = np.mean(np.square(flows_rs2gen - flow_rs2gen_pred))
    res_str = 'Decreased' if prv_diff > new_diff else 'Increased'
    print(f'{res_str} {prv_diff:.5f}=>{new_diff:.5f}')
    wins += (1 if prv_diff > new_diff else 0)
    gens_gs = rectify_img_by_flow_batch(imgs_rs, flow_rs2gen_pred)

    cv2.namedWindow('imgs_gs', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('imgs_gs', imgs_gs[0])
    cv2.namedWindow('imgs_rs', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('imgs_rs', imgs_rs[0])
    cv2.namedWindow('gens_gs', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('gens_gs', gens_gs[0])
    cv2.waitKey()

print (f"Wins: {wins/data_loader.num_train}")



