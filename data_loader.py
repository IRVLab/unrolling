from __future__ import division, absolute_import
import os
import numpy as np
import cv2
import csv
import math
from numpy import linalg as LA

class dataLoader():
    def __init__(self, test_seqs=[2, 9]):
        self.data_path = os.path.join(os.getcwd(), "data/")
        self.img_folder = "cam1/images/"
        self.flow_folder = "cam1/flows_gs2rs/"
        self.depth_folder = "cam1/depth/"
        self.anchor_file = "cam1/poses_cam1_v1.npy"
        self.acc_file = "cam1/acc_t_r.npy"
        train_seqs = [1, 3, 4, 5, 6, 7, 8, 10]

        # get training paths
        self.train_img_paths, self.train_flow_paths, self.train_depth_paths, _ = self.readData(
            train_seqs)
        self.train_anchors = self.readAnchors(train_seqs)

        # get testing paths
        self.test_img_paths, self.test_flow_paths, self.test_depth_paths, self.test_accs = self.readData(
            test_seqs)
        self.test_anchors = self.readAnchors(test_seqs)

    def readData(self, seqs):
        img_paths, flow_paths, depth_paths = [], [], []
        accs = np.empty((0, 6))
        for seq in seqs:
            data_dir = os.path.join(self.data_path, 'seq'+str(seq))
            cur_img_count = os.listdir(
                os.path.join(data_dir, self.flow_folder))
            for fi in range(len(cur_img_count)):
                img_paths.append(os.path.join(
                    data_dir, self.img_folder, str(fi)+'.png'))
                flow_paths.append(os.path.join(
                    data_dir, self.flow_folder, str(fi)+'.npy'))
                depth_paths.append(os.path.join(
                    data_dir, self.depth_folder, str(fi)+'.npy'))

            cur_accs = np.load(os.path.join(data_dir, self.acc_file))
            accs = np.concatenate((accs, cur_accs))
        return img_paths, flow_paths, depth_paths, accs

    def readAnchors(self, seqs):
        anchors = np.empty((0, self.getImgShape()[0], 6))
        for seq in seqs:
            data_dir = os.path.join(self.data_path, 'seq'+str(seq))
            cur_anchors = np.load(os.path.join(data_dir, self.anchor_file))
            anchors = np.concatenate((anchors, cur_anchors))
        return anchors

    def getImgShape(self):
        img = cv2.imread(self.train_img_paths[0], 0)
        return img.shape[:2]

    def loadImg(self, img_paths):
        imgs = []
        for i in range(len(img_paths)):
            img = cv2.imread(img_paths[i], 0)
            imgs.append(img)
        imgs = np.expand_dims(np.array(imgs), -1)
        return imgs

    def loadTrainingImg(self):
        return self.loadImg(self.train_img_paths)

    def loadTestingImg(self):
        return self.loadImg(self.test_img_paths)

    def loadDepth(self, depth_paths):
        depths = []
        for i in range(len(depth_paths)):
            depth = np.load(depth_paths[i])
            depth = np.expand_dims(depth, -1)
            depths.append(depth)
        depths = np.array(depths)
        return depths

    def loadTrainingDepth(self):
        return self.loadDepth(self.train_depth_paths)

    def loadTestingDepth(self):
        return self.loadDepth(self.test_depth_paths)

    def loadFlow(self, flow_paths):
        flows = []
        for i in range(len(flow_paths)):
            flow = np.load(flow_paths[i])
            flows.append(flow)
        flows = np.array(flows)
        return flows

    def loadTrainingFlow(self):
        return self.loadFlow(self.train_flow_paths)

    def loadTestingFlow(self):
        return self.loadFlow(self.test_flow_paths)

    def loadTrainingAnchor(self, num_anchor, rot_weight):
        train_anchors = []
        for i in range(self.train_anchors.shape[0]):
            anchors = []
            for j in range(1, num_anchor+1):
                anchor = self.train_anchors[i][int(
                    j*self.getImgShape()[0]/num_anchor+0.5)-1]
                for k in range(3):
                    anchors.append(anchor[k])
                for k in range(3, 6):
                    anchors.append(rot_weight*anchor[k])
            train_anchors.append(anchors)
        train_anchors = np.array(train_anchors)
        return train_anchors

    def loadTestingAcceleration(self):
        accs = []
        for i in range(self.test_accs.shape[0]):
            acc = self.test_accs[i]
            at = LA.norm(acc[:3])
            ar = LA.norm(acc[3:])
            accs.append([at, ar])
        return np.array(accs)
