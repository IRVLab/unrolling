from __future__ import print_function, division
import os
import cv2
import numpy as np
from data_loader import dataLoader
from unrollnet import UnrollNet

def rectify_img_by_flow(img, flow_gs2rs):
    h, w = img.shape[:2]
    rectified_img = np.zeros_like(img)
    indy, indx = np.indices((h, w), dtype=np.float32)
    map_x = indx.reshape(h, w).astype(np.float32) + flow_gs2rs[:,:,0]
    map_y = indy.reshape(h, w).astype(np.float32) + flow_gs2rs[:,:,1]
    rectified_img = cv2.remap(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), map_x, map_y, cv2.INTER_LINEAR)

    return rectified_img

## dataset and experiment directories
data_dir = os.path.join(os.getcwd(), "data")
seqs = [5] 
data_loader = dataLoader(data_path=data_dir, seqs=seqs) 
ckpt_name = os.path.join(os.getcwd(), "checkpoints/model.hdf5")
model_loader = UnrollNet(data_loader.getImgShape())
model_loader.model.load_weights(ckpt_name)

## training pipeline
wins = 0
for i in range(data_loader.num_train):
    img_rs, flow_gs2rs = data_loader.load(i)
    img_rs_batch = np.expand_dims(img_rs, 0) # (1, h, w, 1)
    flow_gs2rs_pred = model_loader.model.predict(img_rs_batch)[0]

    prv_diff = np.nanmean(np.square(flow_gs2rs))
    new_diff = np.nanmean(np.square(flow_gs2rs - flow_gs2rs_pred))
    res_str = 'Decreased' if prv_diff > new_diff else 'Increased'
    print(f'{res_str} {prv_diff:.5f}=>{new_diff:.5f}')
    wins += (1 if prv_diff > new_diff else 0)

    gens_gs = rectify_img_by_flow(img_rs, flow_gs2rs)
    gens_gs_pred = rectify_img_by_flow(img_rs, flow_gs2rs_pred)

    # cv2.namedWindow('img_rs', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('img_rs', img_rs)
    # cv2.namedWindow('grount-truth', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('grount-truth', gens_gs)
    # cv2.namedWindow('predicted', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('predicted', gens_gs_pred)
    cv2.waitKey()

print (f"Wins: {wins/data_loader.num_train}")



