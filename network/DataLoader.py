# Learning Rolling Shutter Correction from Real Data without Camera Motion Assumption
# Copyright (C) <2021> <Jiawei Mo, Md Jahidul Islam, Junaed Sattar>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import division, absolute_import
import os
import numpy as np
import keras
import cv2


class TumDataSet():
    def __init__(self, gyro=False, acc=False, test_seq=[2, 7], data_path='../data/'):
        num_seqs = 10
        test_seq = np.array(test_seq, dtype='int')

        # parameters
        lut = np.load(os.path.join(data_path, 'seq1/cam1/v1_lut.npy'))
        cam = np.load(os.path.join(data_path, 'seq1/cam1/camera.npy'))
        cam = np.array([[cam[0], 0, cam[2]],
                        [0, cam[1], cam[3]],
                        [0, 0, 1]], dtype='float32')
        self.params = {'lut': lut,
                       'cam': cam,
                       'img_shape': [256, 320]}

        # get entire dataset and split data for training/testing
        self.data = {}
        self.data['train'] = {'size': 0, 'image_paths': [], 'imus': [], 'poses': [],
                              'vels': [], 'depth_paths': [], 'flow_paths': []}
        self.data['val'] = {'size': 0, 'image_paths': [], 'imus': [], 'poses': [],
                            'vels': [], 'depth_paths': [], 'flow_paths': []}
        self.data['test'] = {}
        for seq in range(1, num_seqs+1):
            seq_path = os.path.join(data_path, 'seq'+str(seq)+'/cam1/')
            poses = np.load(seq_path+'pose_cam1_v1.npy')
            vels = poses[:, -1, :]  # velocity = last row pose / 1.0
            imus_raw = np.load(seq_path+'imu_cam1_v1.npy')
            imus = np.empty((imus_raw.shape[0], imus_raw.shape[1], 0))
            if gyro:
                imus = np.append(imus, imus_raw[:, :, :3], axis=-1)
            if acc:
                imus = np.append(imus, imus_raw[:, :, 3:], axis=-1)
            count = len(os.listdir(seq_path + 'flows_rs2gs/'))
            image_paths, depth_paths, flow_paths = [], [], []
            for i in range(count):
                fi = str(i)
                image_paths.append(seq_path + 'images/' + fi + '.png')
                depth_paths.append(seq_path + 'depth/' + fi + '.npy')
                flow_paths.append(seq_path + 'flows_rs2gs/' + str(fi) + '.npy')

            # split data
            if seq in test_seq:
                self.data['test'][seq] = {'size': len(image_paths), 'image_paths': image_paths,
                                          'imus': imus, 'poses': poses, 'vels': vels,
                                          'depth_paths': depth_paths, 'flow_paths': flow_paths}
            else:
                # use the middle 10% as validation
                val_start = int(0.45*count)
                val_end = int(0.55*count)
                self.appendElementByIdx('train', image_paths, imus, depth_paths, flow_paths,
                                        poses, vels, [i for i in range(val_start)])
                self.appendElementByIdx('val', image_paths, imus, depth_paths, flow_paths,
                                        poses, vels, [i for i in range(val_start, val_end)])
                self.appendElementByIdx('train', image_paths, imus, depth_paths, flow_paths,
                                        poses, vels, [i for i in range(val_end, count)])

        self.data['train']['size'] = len(self.data['train']['image_paths'])
        self.data['val']['size'] = len(self.data['val']['image_paths'])

    def appendElementByIdx(self, split, image_paths, imus, depth_paths, flow_paths, poses, vels, indices):
        for i in indices:
            self.data[split]['image_paths'].append(image_paths[i])
            self.data[split]['imus'].append(imus[i])
            self.data[split]['poses'].append(poses[i])
            self.data[split]['vels'].append(vels[i])
            self.data[split]['depth_paths'].append(depth_paths[i])
            self.data[split]['flow_paths'].append(flow_paths[i])


class DataGenerator(keras.utils.Sequence):

    def __init__(self, data, batch_size, dtype):
        self.data = data
        self.batch_size = batch_size
        self.dtype = dtype
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.data['size'] / self.batch_size))

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_image = np.array(
            [cv2.imread(self.data['image_paths'][i]) for i in indexes]) / 255.0
        batch_depth = np.expand_dims(
            np.array([np.load(self.data['depth_paths'][i]) for i in indexes]), axis=-1)
        batch_flow = np.array(
            [np.load(self.data['flow_paths'][i]) for i in indexes])
        batch_depth_flow = np.concatenate((batch_depth, batch_flow), axis=-1)

        if self.dtype == 'depth':
            inputs = batch_image
            outputs = batch_depth
        elif self.dtype == 'vel':
            batch_vel = np.array([self.data['vels'][i] for i in indexes])
            inputs = batch_image
            outputs = {'vel': batch_vel, 'flow': batch_depth_flow}
        else:
            assert(self.dtype == 'pose')
            batch_imu = np.array([self.data['imus'][i] for i in indexes])
            batch_pose = np.array([self.data['poses'][i] for i in indexes])
            inputs = [batch_image, batch_imu]
            outputs = {'pose': batch_pose, 'flow': batch_depth_flow}

        return inputs, outputs

    def on_epoch_end(self):
        self.indexes = np.arange(self.data['size'])
        np.random.shuffle(self.indexes)
