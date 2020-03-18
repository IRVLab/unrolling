from __future__ import division, absolute_import
import os
import numpy as np
import cv2
import csv

class dataLoader():
    def __init__(self, data_path, seq_no=1):
        self.rs_folder = "cam1/images/"
        self.fl_folder = "cam1/flows_gs2rs/"
        self.getTrainPaths(data_path, seq_no)

    def getTrainPaths(self, data_path, seq_no):
        # if None, train all together
        if seq_no not in range(10): 
            all_sets = ["seq"+str(i) for i in range(10)]
        else:
            all_sets = ["seq"+str(seq_no)]
        # get all paths
        self.all_rs_paths, self.all_fl_paths = [], []
        for p in all_sets:
            train_dir = os.path.join(data_path, p)
            rs_paths, fl_paths = self.getPairedPaths(train_dir)
            self.all_rs_paths += rs_paths
            self.all_fl_paths += fl_paths

        self.num_train = len(self.all_fl_paths)
        print (f"Loaded {self.num_train} data for training") 

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

    def loadBatch(self, i, batch_size=1):
        batch_rs = self.all_rs_paths[i*batch_size:(i+1)*batch_size]
        batch_fl = self.all_fl_paths[i*batch_size:(i+1)*batch_size]

        imgs_rs, flows = [], []
        for idx in range(len(batch_rs)): 
            img_rs = cv2.imread(batch_rs[idx], 0).astype(np.float)
            imgs_rs.append(img_rs)
            flow = np.load(batch_fl[idx])
            flows.append(flow)

        imgs_rs = np.array(imgs_rs, dtype=np.uint8)
        flows = np.array(flows, dtype=np.float32)

        imgs_rs = np.expand_dims(imgs_rs, -1) # (b, h, w, 1)
        return imgs_rs, flows








