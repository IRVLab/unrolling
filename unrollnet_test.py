from __future__ import print_function, division
import os
import cv2
import numpy as np
# keras libs
from keras.optimizers import Adam
# local libs
from data_loader import dataLoader
from unrollnet import UnrollNet

def rectify_img_by_flow_batch(imgs, flows_gs2rs):
    b, h, w = imgs.shape[:3]
    rectified_imgs = np.zeros((b,h,w,1), dtype=np.uint8)
    indy, indx = np.indices((h, w), dtype=np.float32)
    map_x = indx.reshape(h, w).astype(np.float32)
    map_y = indy.reshape(h, w).astype(np.float32)
    for bi in range(b):
        map_x_bi = map_x + flows_gs2rs[bi,:,:,0]
        map_y_bi = map_y + flows_gs2rs[bi,:,:,1]
        rectified_img = cv2.remap(cv2.cvtColor(imgs[bi], cv2.COLOR_GRAY2RGB), map_x_bi, map_y_bi, cv2.INTER_LINEAR)
        rectified_imgs[bi,:,:,0] = cv2.cvtColor(rectified_img, cv2.COLOR_RGB2GRAY)

    return rectified_imgs

## dataset and experiment directories
data_dir = os.path.join(os.getcwd(), "data")
seq = 1 # None for all
data_loader = dataLoader(data_path=data_dir, seq_no=seq) 
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
model_loader = UnrollNet(data_loader.getImgShape())
model_loader.model.load_weights(os.path.join(checkpoint_dir, ("model.h5")))

## training pipeline
wins = 0
for i in range(data_loader.num_train):
    imgs_rs, flows_gs2rs = data_loader.loadBatch(i, batch_size=1)
    flows_gs2rs_pred = model_loader.model.predict_on_batch(imgs_rs)

    prv_diff = np.nanmean(np.square(flows_gs2rs))
    new_diff = np.nanmean(np.square(flows_gs2rs - flows_gs2rs_pred))
    res_str = 'Decreased' if prv_diff > new_diff else 'Increased'
    print(f'{res_str} {prv_diff:.5f}=>{new_diff:.5f}')
    wins += (1 if prv_diff > new_diff else 0)

    gens_gs = rectify_img_by_flow_batch(imgs_rs, flows_gs2rs)
    gens_gs_pred = rectify_img_by_flow_batch(imgs_rs, flows_gs2rs_pred)

    cv2.namedWindow('imgs_rs', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('imgs_rs', imgs_rs[0])
    cv2.namedWindow('gens_gs', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('gens_gs', gens_gs[0])
    cv2.namedWindow('gens_gs_pred', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('gens_gs_pred', gens_gs_pred[0])
    cv2.waitKey()

print (f"Wins: {wins/data_loader.num_train}")



