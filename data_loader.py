from __future__ import division, absolute_import
import os
import numpy as np
import cv2
import csv
from numpy import linalg as LA

class dataLoader():
    def __init__(self):
        data_path = os.path.join(os.getcwd(), "data/")
        self.img_folder = "cam1/images/"
        self.flow_folder = "cam1/flows_gs2rs/"
        self.depth_folder = "cam1/depth/"
        self.vel_file = "cam1/vel_t_r.npy"
        self.acc_file = "cam1/acc_t_r.npy"
        self.train_idx = np.load(data_path+'train_idx.npy')
        self.test_idx = np.load(data_path+'test_idx.npy')

        # get all paths
        seqs = [1,2,3,4,5,6,7,8,9,10] 
        self.all_img_paths,self.all_flow_paths,self.all_depth_paths = [],[],[]
        self.all_vels = np.empty((0,6))
        self.all_accs = np.empty((0,6))
        for seq in seqs:
            data_dir = os.path.join(data_path, 'seq'+str(seq))
            img_paths,flow_paths,depth_paths = self.getPaths(data_dir)
            self.all_img_paths += img_paths
            self.all_flow_paths += flow_paths
            self.all_depth_paths += depth_paths
            cur_vels = np.load(os.path.join(data_dir, self.vel_file))
            self.all_vels = np.concatenate((self.all_vels, cur_vels))
            cur_accs = np.load(os.path.join(data_dir, self.vel_file))
            self.all_accs = np.concatenate((self.all_accs, cur_accs))
        
        # order test indices by rolling shutter intensity
        diff_idx = []
        for i in self.test_idx: 
            flow = np.load(self.all_flow_paths[i])
            zero_diff = np.nanmean(np.sqrt(np.sum(np.square(flow), axis=-1)))
            diff_idx.append([zero_diff,i])
        diff_idx = np.array(diff_idx)
        diff_idx = np.array(sorted(diff_idx.tolist(),reverse=True))
        self.test_idx = diff_idx[:,1].astype(int)

        assert((len(self.train_idx)+len(self.test_idx))==len(self.all_depth_paths))

    def getPaths(self, data_dir):
        img_count = os.listdir(os.path.join(data_dir, self.flow_folder))
        img_paths,flow_paths,depth_paths = [],[],[]
        for fi in range(len(img_count)):
            img_paths.append(os.path.join(data_dir, self.img_folder, str(fi)+'.png'))
            flow_paths.append(os.path.join(data_dir, self.flow_folder, str(fi)+'.npy'))
            depth_paths.append(os.path.join(data_dir, self.depth_folder, str(fi)+'.npy'))

        return (img_paths,flow_paths,depth_paths)

    def getImgShape(self):
        img = cv2.imread(self.all_img_paths[0], 0)
        return img.shape[:2]

    def loadByIndices(self, gt_type, indices):
        imgs,gts = [], []
        for i in indices: 
            img = cv2.imread(self.all_img_paths[i], 0)
            imgs.append(img)
            if gt_type == 0:
                flow = np.load(self.all_flow_paths[i])
                gts.append(flow)
            elif gt_type == 1:
                depth = np.load(self.all_depth_paths[i])
                depth = np.expand_dims(depth, -1)
                gts.append(depth)
            else:
                vel = self.all_vels[i]
                vel = np.expand_dims(vel, 0)
                vel = np.expand_dims(vel, 0)
                gts.append(vel)

        imgs = np.expand_dims(np.array(imgs), -1)
        gts = np.array(gts)

        return imgs, gts

    def loadTrainingUnroll(self):
        return self.loadByIndices(0, self.train_idx)

    def loadTestingUnroll(self):
        return self.loadByIndices(0, self.test_idx)

    def loadTrainingDepth(self):
        return self.loadByIndices(1, self.train_idx)

    def loadTrainingVelocity(self):
        return self.loadByIndices(2, self.train_idx)

    def loadTestingDepthVelocity(self):
        _, depth = self.loadByIndices(1, self.test_idx)
        _, velocity = self.loadByIndices(2, self.test_idx)
        return [depth, velocity]

    def loadTestingAcceleration(self):
        accs = []
        for i in self.test_idx: 
            acc = self.all_accs[i]
            at = LA.norm(acc[:3])
            ar = LA.norm(acc[3:])
            accs.append([at,ar])
        return np.array(accs)




