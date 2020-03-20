from __future__ import division, absolute_import
import os
import numpy as np
import cv2
import csv

class dataLoader():
    def __init__(self, data_path, seqs):
        self.rs_folder = "cam1/images/"
        self.fl_folder = "cam1/flows_gs2rs/"
        self.getTrainPaths(data_path, seqs)

    def getTrainPaths(self, data_path, seqs):
        # get all paths
        self.all_rs_paths, self.all_fl_paths = [], []
        for seq in seqs:
            train_dir = os.path.join(data_path, 'seq'+str(seq))
            rs_paths, fl_paths = self.getPairedPaths(train_dir)
            self.all_rs_paths += rs_paths
            self.all_fl_paths += fl_paths

        self.num_train = len(self.all_fl_paths)

    def getPairedPaths(self, data_dir):
        rs_files = os.listdir(os.path.join(data_dir, self.rs_folder))
        rs_paths, fl_paths = [], []
        for fi in range(len(rs_files)):
            rs_paths.append(os.path.join(data_dir, self.rs_folder, str(fi)+'.png'))
            fl_paths.append(os.path.join(data_dir, self.fl_folder, str(fi)+'.npy'))

        return (rs_paths, fl_paths)

    def getImgShape(self):
        img_rs = cv2.imread(self.all_rs_paths[0], 0).astype(np.float)
        return img_rs.shape[:2]

    def load(self, i):
        img_rs = cv2.imread(self.all_rs_paths[i], 0).astype(np.float)
        img_rs = np.array(img_rs, dtype=np.uint8)
        img_rs = np.expand_dims(img_rs, -1) # (h, w, 1)

        flow = np.load(self.all_fl_paths[i])
        
        return img_rs, flow

    def loadAll(self):
        imgs_rs, flows = [], []
        for i in range(len(self.all_rs_paths)): 
            img_rs = cv2.imread(self.all_rs_paths[i], 0).astype(np.float)
            imgs_rs.append(img_rs)
            flow = np.load(self.all_fl_paths[i])
            flows.append(flow)

        imgs_rs = np.array(imgs_rs, dtype=np.uint8)
        imgs_rs = np.expand_dims(imgs_rs, -1) # (b, h, w, 1)

        flows = np.array(flows, dtype=np.float32)

        return imgs_rs, flows








