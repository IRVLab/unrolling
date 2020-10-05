from __future__ import division, absolute_import
import os
import numpy as np
import cv2
import csv
import math
from numpy import linalg as LA


class dataLoader():
    def __init__(self):
        self.trans_weight = 0.3
        self.inverse_depth = False
        self.step = 1
        data_path = os.path.join(os.getcwd(), "data/")
        seqs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # get training paths
        self.img_paths, self.flow_paths, self.depth_paths = [], [], []
        self.accs = np.empty((0, 6))
        for seq in seqs:
            data_dir = os.path.join(data_path, 'seq'+str(seq))
            cur_img_count = os.listdir(
                os.path.join(data_dir, "cam1/flows_gs2rs/"))
            for fi in range(len(cur_img_count)):
                self.img_paths.append(os.path.join(
                    data_dir, "cam1/images/", str(fi)+'.png'))
                self.flow_paths.append(os.path.join(
                    data_dir, "cam1/flows_gs2rs/", str(fi)+'.npy'))
                self.depth_paths.append(os.path.join(
                    data_dir, "cam1/depth/", str(fi)+'.npy'))

            cur_accs = np.load(os.path.join(data_dir, "cam1/acc_t_r.npy"))
            self.accs = np.concatenate((self.accs, cur_accs))

        self.anchors = np.empty((0, self.getImgShape()[0], 6))
        for seq in seqs:
            data_dir = os.path.join(data_path, 'seq'+str(seq))
            cur_anchors = np.load(os.path.join(
                data_dir, "cam1/poses_cam1_v1.npy"))
            self.anchors = np.concatenate((self.anchors, cur_anchors))

        self.train_idx = np.load("data/train_idx.npy")
        self.val_idx = np.load("data/val_idx.npy")
        self.test_idx = np.load("data/test_idx.npy")

    def getImgShape(self):
        img = cv2.imread(self.img_paths[0], 0)
        return img.shape[:2]

    def loadImg(self, idx, grayscale):
        imgs = []
        for i in range(0, idx.shape[0], self.step):
            if grayscale:
                img = cv2.imread(self.img_paths[idx[i]], 0) / 255
                img = np.expand_dims(img, -1)
            else:
                img = cv2.imread(self.img_paths[idx[i]]) / 255
            imgs.append(img)
        return np.array(imgs)

    def loadTrainingImg(self, grayscale=False):
        return self.loadImg(self.train_idx, grayscale)

    def loadValidationImg(self, grayscale=False):
        return self.loadImg(self.val_idx, grayscale)

    def loadTestingImg(self, grayscale=False):
        return self.loadImg(self.test_idx, grayscale)

    def loadDepth(self, idx):
        depths = []
        for i in range(0, idx.shape[0], self.step):
            depth = np.load(self.depth_paths[idx[i]])
            depth = np.expand_dims(depth, -1)
            depths.append(depth)
        depths = np.array(depths)
        if self.inverse_depth:
            depths = np.reciprocal(depths)
        return depths

    def loadTrainingDepth(self):
        return self.loadDepth(self.train_idx)

    def loadValidationDepth(self):
        return self.loadDepth(self.val_idx)

    def loadTestingDepth(self):
        return self.loadDepth(self.test_idx)

    def loadFlow(self, idx):
        flows = []
        for i in range(0, idx.shape[0], self.step):
            flow = np.load(self.flow_paths[idx[i]])
            flows.append(flow)
        flows = np.array(flows)
        return flows

    def loadTrainingFlow(self):
        return self.loadFlow(self.train_idx)

    def loadValidationFlow(self):
        return self.loadFlow(self.val_idx)

    def loadTestingFlow(self):
        return self.loadFlow(self.test_idx)

    def loadAnchor(self, idx, num_anchor):
        anchors = []
        for i in range(0, idx.shape[0], self.step):
            anchors_i = np.empty((num_anchor, 6))
            for ai in range(1, num_anchor+1):
                cur_anchor = self.anchors[idx[i]][int(
                    ai*self.getImgShape()[0]/num_anchor+0.5)-1]
                cur_anchor[:3] = self.trans_weight*cur_anchor[:3]
                anchors_i[ai-1] = cur_anchor
            anchors.append(anchors_i.flatten())
        anchors = np.array(anchors)
        return anchors

    def loadTrainingAnchor(self, num_anchor):
        return self.loadAnchor(self.train_idx, num_anchor)

    def loadValidationAnchor(self, num_anchor):
        return self.loadAnchor(self.val_idx, num_anchor)

    def loadTestingAnchor(self, num_anchor):
        return self.loadAnchor(self.test_idx, num_anchor)

    def loadTestingAcceleration(self):
        accs = []
        for i in range(0, self.test_idx.shape[0], self.step):
            acc = self.accs[self.test_idx[i]]
            at = LA.norm(acc[:3])
            ar = LA.norm(acc[3:])
            accs.append(at*self.trans_weight+ar)
        return np.array(accs)
