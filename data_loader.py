from __future__ import division, absolute_import
import os
import numpy as np
import cv2
import csv
from numpy import linalg as LA


class dataLoader():
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), "data/")
        self.img_folder = "cam1/images/"
        self.flow_folder = "cam1/flows_gs2rs/"
        self.depth_folder = "cam1/depth/"
        self.vel_file = "cam1/vel_t_r.npy"
        self.acc_file = "cam1/acc_t_r.npy"
        train_seqs = [1, 3, 4, 5, 6, 7, 8, 9, 10]
        test_seqs = [2]

        # get training paths
        self.train_img_paths, self.train_flow_paths, self.train_depth_paths = [], [], []
        self.train_vels = np.empty((0, 6))
        self.train_accs = np.empty((0, 6))
        for seq in train_seqs:
            data_dir = os.path.join(self.data_path, 'seq'+str(seq))
            cur_img_count = os.listdir(
                os.path.join(data_dir, self.flow_folder))
            for fi in range(len(cur_img_count)):
                self.train_img_paths.append(os.path.join(
                    data_dir, self.img_folder, str(fi)+'.png'))
                self.train_flow_paths.append(os.path.join(
                    data_dir, self.flow_folder, str(fi)+'.npy'))
                self.train_depth_paths.append(os.path.join(
                    data_dir, self.depth_folder, str(fi)+'.npy'))

            cur_vels = np.load(os.path.join(data_dir, self.vel_file))
            self.train_vels = np.concatenate((self.train_vels, cur_vels))
            cur_accs = np.load(os.path.join(data_dir, self.acc_file))
            self.train_accs = np.concatenate((self.train_accs, cur_accs))

        # get testing paths
        self.test_img_paths, self.test_flow_paths, self.test_depth_paths = [], [], []
        self.test_vels = np.empty((0, 6))
        self.test_accs = np.empty((0, 6))
        for seq in test_seqs:
            data_dir = os.path.join(self.data_path, 'seq'+str(seq))
            cur_img_count = os.listdir(
                os.path.join(data_dir, self.flow_folder))
            for fi in range(len(cur_img_count)):
                self.test_img_paths.append(os.path.join(
                    data_dir, self.img_folder, str(fi)+'.png'))
                self.test_flow_paths.append(os.path.join(
                    data_dir, self.flow_folder, str(fi)+'.npy'))
                self.test_depth_paths.append(os.path.join(
                    data_dir, self.depth_folder, str(fi)+'.npy'))

            cur_vels = np.load(os.path.join(data_dir, self.vel_file))
            self.test_vels = np.concatenate((self.test_vels, cur_vels))
            cur_accs = np.load(os.path.join(data_dir, self.acc_file))
            self.test_accs = np.concatenate((self.test_accs, cur_accs))

    def getImgShape(self):
        img = cv2.imread(self.train_img_paths[0], 0)
        return img.shape[:2]

    def loadTrainingUnroll(self):
        imgs, flows = [], []
        for i in range(len(self.train_img_paths)):
            img = cv2.imread(self.train_img_paths[i], 0)
            imgs.append(img)
            flow = np.load(self.train_flow_paths[i])
            flows.append(flow)
        imgs = np.expand_dims(np.array(imgs), -1)
        flows = np.array(flows)
        return imgs, flows

    def loadTestingUnroll(self):
        imgs, flows = [], []
        for i in range(len(self.test_img_paths)):
            img = cv2.imread(self.test_img_paths[i], 0)
            imgs.append(img)
            flow = np.load(self.test_flow_paths[i])
            flows.append(flow)
        imgs = np.expand_dims(np.array(imgs), -1)
        flows = np.array(flows)
        return imgs, flows

    def loadTrainingDepth(self):
        imgs, depths = [], []
        for i in range(len(self.train_img_paths)):
            img = cv2.imread(self.train_img_paths[i], 0)
            imgs.append(img)
            depth = np.load(self.train_depth_paths[i])
            depth = np.expand_dims(depth, -1)
            depths.append(depth)
        imgs = np.expand_dims(np.array(imgs), -1)
        depths = np.array(depths)
        return imgs, depths

    def loadTrainingVelocity(self):
        imgs, vels = [], []
        for i in range(len(self.train_img_paths)):
            img = cv2.imread(self.train_img_paths[i], 0)
            imgs.append(img)
            vel = self.train_vels[i]
            vel = np.expand_dims(vel, 0)
            vel = np.expand_dims(vel, 0)
            vels.append(vel)
        imgs = np.expand_dims(np.array(imgs), -1)
        vels = np.array(vels)
        return imgs, vels

    def loadTestingDepthVelocity(self):
        depths, vels = [], []
        for i in range(len(self.test_depth_paths)):
            depth = cv2.imread(self.test_depth_paths[i], 0)
            depth = np.expand_dims(depth, -1)
            depths.append(depth)
            vel = self.test_vels[i]
            vel = np.expand_dims(vel, 0)
            vel = np.expand_dims(vel, 0)
            vels.append(vel)
        depths = np.array(depths)
        vels = np.array(vels)
        return depths, vels

    def loadTestingAcceleration(self):
        accs = []
        for i in self.test_accs:
            acc = self.test_accs[i]
            at = LA.norm(acc[:3])
            ar = LA.norm(acc[3:])
            accs.append([at, ar])
        return np.array(accs)

    # def loadTestRSIntensity(self):
    #     for i in self.test_flow_paths:
    #         flow = np.load(self.test_flow_paths[i])
    #         zero_diff = np.nanmean(np.sqrt(np.sum(np.square(flow), axis=-1)))
    #         test_rs_intensity.append([zero_diff, i])
    #     return np.array(test_rs_intensity)
