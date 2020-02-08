#!/usr/bin/env python
"""
# > Various modules for handling data 
#
# Maintainer: Jahid (email: islam034@umn.edu)
# Interactive Robotics and Vision Lab (http://irvlab.cs.umn.edu/)
# Any part of this repo can be used for academic and educational purposes only
"""
from __future__ import division
from __future__ import absolute_import
import os
import random
import fnmatch
import numpy as np
import cv2

def deprocess(x):
    # [-1,1] -> [0, 255]
    return (x+1.0)*0.5*255.


def preprocess(x):
    # [0,255] -> [-1, 1]
    return (x/127.5)-1.0


def getPaths(data_dir):
    exts = ['*.png','*.PNG','*.jpg','*.JPG', '*.JPEG']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if (fnmatch.fnmatch(filename, pattern)):
                    fname_ = os.path.join(d,filename)
                    image_paths.append(fname_)
    return np.asarray(image_paths)


def read_and_resize(paths, res=(480, 640), mode_='RGB'):
    img = cv2.imread(paths, 0).astype(np.float)
    img = cv2.resize(img, res)
    return img


class dataLoaderTUM():
    def __init__(self, data_path="/mnt/data2/unRolling_Shutter/Rolling_Shutter_TUM/", seq_no=1, res=(320, 256)):
        self.gs_folder = "img0/"
        self.rs_folder = "img1/" 
        self.res_ = res
        self.get_train_paths(data_path, seq_no)

    def get_train_paths(self, data_path, seq_no):
        # if None, train all together
        if seq_no not in range(10): 
            all_sets = ["seq"+str(i) for i in range(10)]
        else:
            all_sets = ["seq"+str(seq_no)]
        # get all paths
        self.all_gs_paths, self.all_rs_paths = [], []
        for p in all_sets:
            train_dir = os.path.join(data_path, p)
            gs_path, rs_path = self.get_paired_paths(train_dir)
            self.all_gs_paths += gs_path
            self.all_rs_paths += rs_path
        self.num_train = len(self.all_rs_paths)
        print ("Loaded {0} pairs of image-paths for training".format(self.num_train)) 

    def get_paired_paths(self, data_dir):
        gs_path = sorted(os.listdir(os.path.join(data_dir, self.gs_folder)))   
        rs_path = sorted(os.listdir(os.path.join(data_dir, self.rs_folder)))
        num_paths = min(len(gs_path), len(rs_path))
        all_gs_paths, all_rs_paths = [], []
        for f in gs_path[:num_paths]:
            all_gs_paths.append(os.path.join(data_dir, self.gs_folder, f))
            all_rs_paths.append(os.path.join(data_dir, self.rs_folder, f))
        return (all_gs_paths, all_rs_paths)


    def load_batch(self, batch_size=1):
        self.n_batches = self.num_train//batch_size
        for i in range(self.n_batches-1):
            batch_gs = self.all_gs_paths[i*batch_size:(i+1)*batch_size]
            batch_rs = self.all_rs_paths[i*batch_size:(i+1)*batch_size]
            # yeild batch
            imgs_gs, imgs_rs = [], []
            for idx in range(len(batch_gs)): 
                img_gs = read_and_resize(batch_gs[idx], res=self.res_)
                img_rs = read_and_resize(batch_rs[idx], res=self.res_)
                imgs_gs.append(img_gs)
                imgs_rs.append(img_rs)
            imgs_gs = preprocess(np.array(imgs_gs))
            imgs_rs = preprocess(np.array(imgs_rs))
            #return imgs_gs, imgs_rs
            imgs_gs = np.expand_dims(imgs_gs, -1) # (b, h, w, 1)
            imgs_rs = np.expand_dims(imgs_rs, -1) # (b, h, w, 1)
            #print imgs_gs.shape, imgs_rs.shape
            yield imgs_gs, imgs_rs



if __name__=="__main__":
    data_loader = dataLoaderTUM()
    imgs_gs, imgs_rs = data_loader.load_batch()
    cv2.imwrite('a.png', deprocess(imgs_gs[0,:]))






