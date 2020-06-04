from __future__ import absolute_import, division, print_function
from keras.models import Input, Model
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, Concatenate, UpSampling2D
import tensorflow as tf

def iconv_pr(input, pr_prv, conv, filters, lvl):
    upconv = Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same', name='upconv'+str(lvl))(input)
    upconv = Activation('relu')(BatchNormalization(axis=3, name='upconv{}bn'.format(lvl))(upconv))
    pr_prv_upsampled = UpSampling2D(name='prupsample'+str(lvl))(pr_prv)
    upconv_prprv_conv = Concatenate(axis=3)([upconv, pr_prv_upsampled, conv])
    iconv = Activation('relu')(Conv2D(filters, kernel_size=3, strides=1, padding='same', name='iconv'+str(lvl))(upconv_prprv_conv))
    pr = Activation('relu')(Conv2D(1, kernel_size=3, strides=1, padding='same', name='pr'+str(lvl))(iconv))
    return [iconv, pr]

def DispNet(img_input):
    # encoder
    conv1 = Activation('relu')(Conv2D(64, kernel_size=7, strides=2, padding='same', name='conv1')(img_input))
    conv2 = Activation('relu')(Conv2D(128, kernel_size=5, strides=2, padding='same', name='conv2')(conv1))
    conv3a = Activation('relu')(Conv2D(256, kernel_size=5, strides=2, padding='same', name='conv3a')(conv2))
    conv3b = Activation('relu')(Conv2D(256, kernel_size=3, strides=1, padding='same', name='conv3b')(conv3a))
    conv4a = Activation('relu')(Conv2D(512, kernel_size=3, strides=2, padding='same', name='conv4a')(conv3b))
    conv4b = Activation('relu')(Conv2D(512, kernel_size=3, strides=1, padding='same', name='conv4b')(conv4a))
    conv5a = Activation('relu')(Conv2D(512, kernel_size=3, strides=2, padding='same', name='conv5a')(conv4b))
    conv5b = Activation('relu')(Conv2D(512, kernel_size=3, strides=1, padding='same', name='conv5b')(conv5a))
    conv6a = Activation('relu')(Conv2D(1024, kernel_size=3, strides=2, padding='same', name='conv6a')(conv5b))
    conv6b = Activation('relu')(Conv2D(1024, kernel_size=3, strides=1, padding='same', name='conv6b')(conv6a))

    pr6 = Activation('relu')(Conv2D(1, kernel_size=3, strides=1, padding='same', name='pr6')(conv6a))

    # decoder
    [iconv5, pr5] = iconv_pr(conv6b, pr6, conv5b, 512, 5) 
    [iconv4, pr4] = iconv_pr(iconv5, pr5, conv4b, 256, 4)
    [iconv3, pr3] = iconv_pr(iconv4, pr4, conv3b, 128, 3)
    [iconv2, pr2] = iconv_pr(iconv3, pr3, conv2, 64, 2)
    [iconv1, pr1] = iconv_pr(iconv2, pr2, conv1, 32, 1)

    return [pr6, pr5, pr4, pr3, pr2, pr1]


class DepthNet():
    def __init__(self, im_shape):
        img = Input(shape=(im_shape[0], im_shape[1], 1))
        [pr6, pr5, pr4, pr3, pr2, pr1] = DispNet(img)
        depth = UpSampling2D(name='y_pred')(pr1)
        self.model = Model(input=img, output=depth)

    def depthLoss(self, y_true, y_pred):
        diff = y_true-y_pred
        mask = tf.math.is_nan(diff)
        count = tf.reduce_sum(1 - tf.cast(mask, diff.dtype))
        diff = tf.where(mask, tf.zeros_like(diff), diff)
        mean = tf.reduce_sum(tf.square(diff)) / count
        return mean
