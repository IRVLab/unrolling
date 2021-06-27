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

# fmt: off
from __future__ import absolute_import, division, print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from keras.models import Input, Model
from keras.layers import Conv2D, Conv2DTranspose, Activation, LSTM
from keras.layers import BatchNormalization, Concatenate, Reshape
import numpy as np

from helpers import baseNet, flowLossByPose
# fmt: on


def poseConv(x, filters, kernel_size, lvl):
    x = Conv2D(filters, kernel_size, (1, 2), 'same',
               activation='relu', name='conv'+str(lvl))(x)
    x = Conv2DTranspose(filters, kernel_size, (2, 1),
                        'same', name='upconv'+str(lvl))(x)
    x = Activation('relu')(BatchNormalization(name='bn'+str(lvl))(x))
    return x


class RsPoseNet():
    def __init__(self, params, gyro=True, acc=True):
        rows, cols = params['img_shape']
        self.params = params
        self.pose_scale = 10

        # inputs
        input_img = Input((rows, cols, 3))
        input_imu = Input((rows, 3*gyro+3*acc))
        pose_scaled = self.poseNet(input_img, input_imu)  # for poseLoss
        pose_mat = self.convertToPoseMat(pose_scaled)  # for flowLoss

        self.model = Model(inputs=[input_img, input_imu], outputs={
                           'pose': pose_scaled, 'flow': pose_mat})

    def poseNet(self, input_img, input_imu):
        features = baseNet(input_img)

        x = poseConv(features, 512, 3, 5)
        x = poseConv(x, 256, 3, 4)
        x = poseConv(x, 128, 3, 3)
        x = poseConv(x, 64, 3, 2)
        x = poseConv(x, 32, 3, 1)

        img_pose = Conv2D(6, 1, 1, 'same', activation='tanh',
                          name='pose_conv')(x)

        img_pose = Reshape(
            (self.params['img_shape'][0], 6), name='img_pose')(img_pose)

        # extend pose from image by IMU
        img_imu = Concatenate()([img_pose, input_imu])
        h = LSTM(6, return_sequences=True, name='imu_lstm1')(img_imu)
        pose = LSTM(6, return_sequences=True, name='pose')(h)

        return pose

    def convertToPoseMat(self, pose_scaled):
        # -1 x 256 x 6 -> -1 x 256 x 320 x 6
        pose_mat = tf.gather(pose_scaled / self.pose_scale,
                             self.params['lut'], axis=1, name='flow')
        return pose_mat

    def poseLoss(self, y_true, y_pred):
        return tf.reduce_mean(tf.norm(self.pose_scale*y_true - y_pred, axis=-1))

    def flowLoss(self, y_true, y_pred):
        return flowLossByPose(y_true, y_pred, self.params)
