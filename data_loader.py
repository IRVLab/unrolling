from __future__ import division, absolute_import
import os
import numpy as np
import cv2
import csv

class dataLoader():
    def __init__(self):
        data_path = os.path.join(os.getcwd(), "data/")
        self.img_folder = "cam1/images/"
        self.flow_folder = "cam1/flows_gs2rs/"
        self.train_idx = np.load(data_path+'train_idx.npy')
        self.test_idx = np.load(data_path+'test_idx.npy')

        # get all paths
        seqs = [1,2,3,4,5,6,7,8,9,10] 
        self.all_img_paths,self.all_flow_paths = [],[]
        for seq in seqs:
            data_dir = os.path.join(data_path, 'seq'+str(seq))
            img_paths,flow_paths = self.getPairedPaths(data_dir)
            self.all_img_paths += img_paths
            self.all_flow_paths += flow_paths
        
        assert(len(self.all_img_paths)==len(self.all_flow_paths))
        assert((len(self.train_idx)+len(self.test_idx))==len(self.all_img_paths))

    def getPairedPaths(self, data_dir):
        img_files = os.listdir(os.path.join(data_dir, self.img_folder))
        img_paths,flow_paths = [],[]
        for fi in range(len(img_files)):
            img_paths.append(os.path.join(data_dir, self.img_folder, str(fi)+'.png'))
            flow_paths.append(os.path.join(data_dir, self.flow_folder, str(fi)+'.npy'))

        return (img_paths,flow_paths)

    def getImgShape(self):
        img_rs = cv2.imread(self.all_img_paths[0], 0)
        return img_rs.shape[:2]

    def load(self, i):
        img_rs = cv2.imread(self.all_img_paths[i], 0)
        flow = np.load(self.all_flow_paths[i])

        return img_rs, flow

    def loadAll(self):
        imgs_rs,flows = [], []
        for i in range(len(self.all_img_paths)): 
            img_rs,flow = self.load(i)
            imgs_rs.append(img_rs)
            flows.append(flow)

        imgs_rs = np.expand_dims(np.array(imgs_rs), -1)
        flows = np.array(flows)

        return imgs_rs, flows

    def loadTraining(self):
        imgs_rs,flows = [], []
        for i in self.train_idx: 
            img_rs,flow = self.load(i)
            imgs_rs.append(img_rs)
            flows.append(flow)

        imgs_rs = np.expand_dims(np.array(imgs_rs), -1)
        flows = np.array(flows)

        return imgs_rs, flows

    def loadTesting(self):
        imgs_rs,flows = [], []
        for i in self.test_idx: 
            img_rs,flow = self.load(i)
            imgs_rs.append(img_rs)
            flows.append(flow)

        imgs_rs = np.expand_dims(np.array(imgs_rs), -1)
        flows = np.array(flows)

        return imgs_rs, flows





