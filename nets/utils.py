#!/usr/bin/env python
"""
# > Various modules for handling data 
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
from __future__ import division, absolute_import
import os
import random
import fnmatch
import numpy as np
import cv2
import tensorflow as tf
from keras import backend as K

def draw_flow(img, flow):
    h, w = img.shape[:2]
    flow_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # edges = cv2.Canny(img, 20, 50)
    for hi in range(h):
        for wi in range(w):
            wf,hf = flow[hi,wi].T
            # if edges[hi, wi]: 
            if hf>0.2: 
                flow_vis[hi, wi] = (0, 127*hf, 0)
            if hf<-0.2: 
                flow_vis[hi, wi] = (0, 0, -127*hf)

    return flow_vis

def draw_img_by_flow_batch(imgs, flows):
    b, h, w = imgs.shape[:3]
    flow_imgs = np.zeros((b,h,w), dtype=np.uint8)
    indy, indx = np.indices((h, w), dtype=np.float32)
    map_x = indx.reshape(h, w).astype(np.float32)
    map_y = indy.reshape(h, w).astype(np.float32)
    for bi in range(b):
        map_x_bi = map_x - flows[bi,:,:,0]
        map_y_bi = map_y - flows[bi,:,:,1]
        flow_img = cv2.remap(cv2.cvtColor(imgs[bi], cv2.COLOR_GRAY2RGB), map_x_bi, map_y_bi, cv2.INTER_LINEAR)
        flow_imgs[bi,:,:] = cv2.cvtColor(flow_img, cv2.COLOR_RGB2GRAY)

    return flow_imgs


class dataLoaderTUM():
    def __init__(self, data_path="/mnt/data2/unRolling_Shutter/Rolling_Shutter_TUM/", seq_no=1, res=(320, 256), load_flow=False):
        self.gs_folder = "img0/"
        self.rs_folder = "img1/" 
        self.fl_folder = "flow/" 
        self.load_flow = load_flow
        self.res = res
        self.get_train_paths(data_path, seq_no)

    def get_train_paths(self, data_path, seq_no):
        # if None, train all together
        if seq_no not in range(10): 
            all_sets = ["seq"+str(i) for i in range(10)]
        else:
            all_sets = ["seq"+str(seq_no)]
        # get all paths
        self.all_gs_paths, self.all_rs_paths, self.all_fl_paths = [], [], []
        for p in all_sets:
            train_dir = os.path.join(data_path, p)
            gs_path, rs_path, fl_path = self.get_paired_paths(train_dir)
            self.all_gs_paths += gs_path
            self.all_rs_paths += rs_path
            self.all_fl_paths += fl_path
        self.num_train = len(self.all_rs_paths)
        print ("Loaded {0} pairs of image-paths for training".format(self.num_train)) 

    def get_paired_paths(self, data_dir):
        gs_path = sorted(os.listdir(os.path.join(data_dir, self.gs_folder)))   
        rs_path = sorted(os.listdir(os.path.join(data_dir, self.rs_folder)))
        num_paths = min(len(gs_path), len(rs_path))
        if self.load_flow:
            fl_path = sorted(os.listdir(os.path.join(data_dir, self.fl_folder)))
            num_paths = min(num_paths, len(fl_path))
        all_gs_paths, all_rs_paths, all_fl_paths = [], [], []
        for f in gs_path[:num_paths]:
            all_gs_paths.append(os.path.join(data_dir, self.gs_folder, f))
            all_rs_paths.append(os.path.join(data_dir, self.rs_folder, f))
        if self.load_flow:
            for f in fl_path[:num_paths]:
                all_fl_paths.append(os.path.join(data_dir, self.fl_folder, f))
        return (all_gs_paths, all_rs_paths, all_fl_paths)

    def read_and_resize(self, paths):
        img = cv2.imread(paths, 0).astype(np.float)
        img = cv2.resize(img, self.res)
        return img

    def load_batch(self, i, batch_size=1):
        batch_gs = self.all_gs_paths[i*batch_size:(i+1)*batch_size]
        batch_rs = self.all_rs_paths[i*batch_size:(i+1)*batch_size]
        if self.load_flow:
            batch_fl = self.all_fl_paths[i*batch_size:(i+1)*batch_size]
        # yeild batch
        imgs_gs, imgs_rs, flows = [], [], []
        for idx in range(len(batch_gs)): 
            img_gs = self.read_and_resize(batch_gs[idx])
            img_rs = self.read_and_resize(batch_rs[idx])
            imgs_gs.append(img_gs)
            imgs_rs.append(img_rs)
            if self.load_flow:
                flow = np.load(batch_fl[idx])
                flows.append(flow)
        imgs_gs = np.array(imgs_gs, dtype=np.uint8)
        imgs_rs = np.array(imgs_rs, dtype=np.uint8)
        flows = np.array(flows, dtype=np.float32)
        #return imgs_gs, imgs_rs
        imgs_gs = np.expand_dims(imgs_gs, -1) # (b, h, w, 1)
        imgs_rs = np.expand_dims(imgs_rs, -1) # (b, h, w, 1)
        #print imgs_gs.shape, imgs_rs.shape
        return imgs_gs, imgs_rs, flows



if __name__=="__main__":
    data_loader = dataLoaderTUM()
    imgs_gs, imgs_rs = data_loader.load_batch()
    cv2.imwrite('a.png', imgs_gs[0,:])






