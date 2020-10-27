from __future__ import division, absolute_import
import os
import numpy as np
import cv2
import csv
import math
from numpy import linalg as LA


class dataLoader():
    def __init__(self):
        data_path = '/mnt/data2/jiawei/unrolling/data/'
        self.step = 1
        num_seqs = 10
        test_seq = 2

        # get training paths
        self.img_paths, self.flow_paths, self.depth_paths = [], [], []
        self.accs = np.empty((0, 6))
        for seq in range(1, num_seqs+1):
            seq_path = os.path.join(data_path, 'seq'+str(seq))
            cur_img_count = len(os.listdir(
                os.path.join(seq_path, "cam1/flows_gs2rs/")))
            for fi in range(cur_img_count):
                self.img_paths.append(os.path.join(
                    seq_path, "cam1/images/", str(fi)+'.png'))
                self.flow_paths.append(os.path.join(
                    seq_path, "cam1/flows_gs2rs/", str(fi)+'.npy'))
                self.depth_paths.append(os.path.join(
                    seq_path, "cam1/depth/", str(fi)+'.npy'))

            cur_accs = np.load(os.path.join(seq_path, "cam1/acc_t_r.npy"))
            self.accs = np.concatenate((self.accs, cur_accs))

        self.anchors = np.empty((0, self.getImgShape()[0], 6))
        for seq in range(1, num_seqs+1):
            seq_path = os.path.join(data_path, 'seq'+str(seq))
            cur_anchors = np.load(os.path.join(
                seq_path, "cam1/poses_cam1_v1.npy"))
            self.anchors = np.concatenate((self.anchors, cur_anchors))

        # indices
        self.train_idx = np.load(os.path.join(data_path, 'train_idx.npy'))
        self.val_idx = np.load(os.path.join(data_path, 'val_idx.npy'))
        self.test_idx = np.load(os.path.join(data_path, 'test_idx.npy'))

        # seq2 data for ground-truth verification
        seq_img_start = 0
        for seq in range(1, test_seq):
            seq_path = os.path.join(data_path, 'seq'+str(seq))
            cur_img_count = len(os.listdir(
                os.path.join(seq_path, "cam1/flows_gs2rs/")))
            seq_img_start += cur_img_count
        seq_path = os.path.join(data_path, 'seq'+str(test_seq))
        seq_img_count = len(os.listdir(
            os.path.join(seq_path, "cam1/flows_gs2rs/")))
        self.seq_idx = np.arange(
            seq_img_start, seq_img_start+seq_img_count)

        # camera parameters
        self.cam = np.load(os.path.join(data_path, 'seq1/cam1/camera.npy'))

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

    def loadSeqImg(self, grayscale=False):
        return self.loadImg(self.seq_idx, grayscale)

    def loadDepth(self, idx):
        depths = []
        for i in range(0, idx.shape[0], self.step):
            depth = np.load(self.depth_paths[idx[i]])
            depth = np.expand_dims(depth, -1)
            depths.append(depth)
        depths = np.array(depths)
        return depths

    def loadTrainingDepth(self):
        return self.loadDepth(self.train_idx)

    def loadValidationDepth(self):
        return self.loadDepth(self.val_idx)

    def loadTestingDepth(self):
        return self.loadDepth(self.test_idx)

    def loadSeqDepth(self):
        return self.loadDepth(self.seq_idx)

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

    def loadSeqFlow(self):
        return self.loadFlow(self.seq_idx)

    def loadAnchor(self, idx, num_anchor):
        anchor_idx = []
        for ai in range(1, num_anchor+1):
            anchor_idx.append(int(ai*self.getImgShape()[0]/num_anchor+0.5)-1)

        anchors = []
        for i in range(0, idx.shape[0], self.step):
            anchors.append(self.anchors[idx[i]][anchor_idx].flatten())
        anchors = np.array(anchors)
        # rearrange to [t0, t1, ..., r1, r2] for the weight in the loss function
        anchors_t_r = np.empty_like(anchors)
        for ai in range(num_anchor):
            anchors_t_r[:, (3*ai):(3*ai+3)] = anchors[:, (6*ai):(6*ai+3)]
            anchors_t_r[:, (3*num_anchor+3*ai):(3*num_anchor +
                                                3*ai+3)] = anchors[:, (6*ai+3):(6*ai+6)]
        return anchors_t_r

    def loadTrainingAnchor(self, num_anchor):
        return self.loadAnchor(self.train_idx, num_anchor)

    def loadValidationAnchor(self, num_anchor):
        return self.loadAnchor(self.val_idx, num_anchor)

    def loadTestingAnchor(self, num_anchor):
        return self.loadAnchor(self.test_idx, num_anchor)

    def loadSeqAnchor(self, num_anchor):
        return self.loadAnchor(self.seq_idx, num_anchor)

    def loadAcceleration(self, idx):
        accs = []
        for i in range(0, idx.shape[0], self.step):
            acc = self.accs[idx[i]]
            at = LA.norm(acc[:3])
            ar = LA.norm(acc[3:])
            accs.append(0.3*at+ar)
        return np.array(accs)

    def loadTestingAcceleration(self):
        return self.loadAcceleration(self.test_idx)

    def loadSeqAcceleration(self):
        return self.loadAcceleration(self.seq_idx)
