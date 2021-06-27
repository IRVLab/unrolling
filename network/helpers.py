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

from __future__ import absolute_import, division, print_function
import tensorflow as tf
from classification_models.keras import Classifiers
from keras.applications.vgg16 import VGG16
import numpy as np


def getFlow(depth, pose, params):
    rows, cols = params['img_shape']

    # split pose to translation and angel-axis representation rotation
    trans, rot = tf.split(pose, [3, 3], -1)
    angle = tf.expand_dims(tf.norm(rot, axis=-1), -1)
    cos_angle = tf.cos(angle)
    axis = rot / (angle + 1e-20)

    # recover 3D point in RS frame
    v_rs, u_rs = np.indices((rows, cols), dtype='float32')
    Kiu_rs = np.matmul(np.stack([u_rs, v_rs, np.ones_like(
        u_rs)], axis=-1), np.linalg.inv(params['cam']).T)
    depth = tf.stack([tf.squeeze(depth, axis=-1)]*3, axis=-1)
    xyz_rs = depth * Kiu_rs

    # project to GS frame
    xyz_gs = xyz_rs * cos_angle + tf.linalg.cross(axis, xyz_rs) * tf.sin(
        angle) + axis * tf.reduce_sum(axis * xyz_rs, -1, True) * (1-cos_angle) + trans
    uvd_gs = tf.matmul(xyz_gs, params['cam'].T)
    uv_gs, d_gs = tf.split(uvd_gs, [2, 1], -1)
    uv_gs = uv_gs / (d_gs + 1e-20)

    uv_rs = np.stack([u_rs, v_rs], axis=-1)
    flow = tf.identity(uv_gs - uv_rs, name='flow')
    return flow


def flowLossByPose(df_true, p_pred, params):
    d_true, f_true = tf.split(df_true, [1, 2], -1)
    d_true = tf.where(tf.math.is_nan(d_true),
                      tf.ones_like(d_true), d_true)
    f_pred = getFlow(d_true, p_pred, params)
    diff = tf.where(tf.math.is_nan(f_true),
                    tf.zeros_like(f_true), f_true-f_pred)
    return tf.reduce_mean(tf.norm(diff, axis=-1))


def baseNet(img_input, base='ResNet34'):
    _, rows, cols, _ = img_input.shape
    if base == 'ResNet34':
        ResNet34, _ = Classifiers.get('resnet34')
        features = ResNet34(input_shape=(rows, cols, 3),
                            weights='imagenet', include_top=False)(img_input)
    elif base == 'ResNet50':
        ResNet50, _ = Classifiers.get('resnet50')
        features = ResNet50(input_shape=(rows, cols, 3),
                            weights='imagenet', include_top=False)(img_input)
    elif base == 'VGG16':
        vgg = VGG16(input_shape=(rows, cols, 3),
                    weights='imagenet', include_top=False)(img_input)
        features = vgg.get_layer('block5_pool')
    return features
