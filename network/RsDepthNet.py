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
from keras.layers import Conv2D, Conv2DTranspose, Activation
from keras.layers import BatchNormalization, UpSampling2D, Concatenate

# fmt: on


def iconv_pr(iconv_prv, pr_prv, conv_encoder, filters, lvl):
    upconv = Activation('relu')(BatchNormalization(name='upconv{}bn'.format(lvl))(
        Conv2DTranspose(filters, 4, 2, 'same', name='upconv'+str(lvl))(iconv_prv)))
    upsampled = UpSampling2D(interpolation='bilinear',
                             name='upsampled'+str(lvl))(pr_prv)
    ctnt = Concatenate()([upconv, upsampled, conv_encoder])
    iconv = Conv2D(filters, 3, 1, 'same', activation='relu',
                   name='iconv'+str(lvl))(ctnt)
    pr = Conv2D(1, 3, 1, 'same', activation='relu',
                name='pr'+str(lvl))(iconv)
    return [iconv, pr]


class RsDepthNet():
    def __init__(self, params):
        rows, cols = params['img_shape']
        input_img = Input((rows, cols, 3))
        depth = self.depthNet(input_img)
        self.model = Model(inputs=input_img, outputs=depth)

    def depthNet(self, img):
        # encoder
        c1 = Conv2D(64, 7, 2, 'same', activation='relu', name='conv1')(img)
        c2 = Conv2D(128, 5, 2, 'same', activation='relu', name='conv2')(c1)
        c3a = Conv2D(256, 5, 2, 'same', activation='relu', name='conv3a')(c2)
        c3b = Conv2D(256, 3, 1, 'same', activation='relu', name='conv3b')(c3a)
        c4a = Conv2D(512, 3, 2, 'same', activation='relu', name='conv4a')(c3b)
        c4b = Conv2D(512, 3, 1, 'same', activation='relu', name='conv4b')(c4a)
        c5a = Conv2D(512, 3, 2, 'same', activation='relu', name='conv5a')(c4b)
        c5b = Conv2D(512, 3, 1, 'same', activation='relu', name='conv5b')(c5a)
        c6a = Conv2D(1024, 3, 2, 'same', activation='relu', name='conv6a')(c5b)
        c6b = Conv2D(1024, 3, 1, 'same', activation='relu', name='conv6b')(c6a)

        pr6 = Conv2D(1, 3, 1, 'same', activation='relu', name='pr6')(c6a)

        # decoder
        [ic5, pr5] = iconv_pr(c6b, pr6, c5b, 512, 5)
        [ic4, pr4] = iconv_pr(ic5, pr5, c4b, 256, 4)
        [ic3, pr3] = iconv_pr(ic4, pr4, c3b, 128, 3)
        [ic2, pr2] = iconv_pr(ic3, pr3, c2, 64, 2)
        [ic1, pr1] = iconv_pr(ic2, pr2, c1, 32, 1)

        d1 = UpSampling2D(interpolation='bilinear', name='depth')(pr1)

        return d1

    def depthLoss(self, y_true, y_pred):
        diff = tf.where(tf.math.is_nan(y_true),
                        tf.zeros_like(y_true), y_true-y_pred)
        return tf.reduce_mean(tf.abs(diff))
