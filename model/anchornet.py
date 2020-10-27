from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten
from classification_models.keras import Classifiers
from keras.applications.vgg16 import VGG16


def Conv2d_BN_Relu(x, filters, kernel_size, name, strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding,
               strides=strides, name=name+'_conv')(x)
    x = BatchNormalization(axis=3, name=name+'_bn')(x)
    x = Activation('relu')(x)
    return x


class AnchorNet():
    def __init__(self, im_shape, num_anchor, base='ResNet34'):
        self.num_anchor = num_anchor

        if base == 'ResNet34':
            ResNet34, _ = Classifiers.get('resnet34')
            resnet = ResNet34(input_shape=(im_shape[0], im_shape[1], 3),
                              weights='imagenet', include_top=False)
            input, features = resnet.input, resnet.output
        elif base == 'VGG16':
            vgg = VGG16(input_shape=(im_shape[0], im_shape[1], 3),
                        include_top=False, weights='imagenet')
            input, features = vgg.input, vgg.get_layer('block5_pool').output

        vel = self.AnchorNet(features)
        self.model = Model(inputs=input, outputs=vel)

    def AnchorNet(self, input):
        # AnchorNet
        x = Conv2d_BN_Relu(input, filters=512,
                           kernel_size=(3, 3), name='V0')
        x = Conv2d_BN_Relu(x, filters=256, kernel_size=(3, 3), name='V1')
        x = Conv2d_BN_Relu(x, filters=128, kernel_size=(3, 3), name='V2')
        x = Conv2d_BN_Relu(x, filters=64, kernel_size=(3, 3), name='V3')
        x = Conv2d_BN_Relu(x, filters=32, kernel_size=(3, 3), name='V4')

        x_shape = x.get_shape().as_list()
        anchors = Conv2D(
            6*self.num_anchor, kernel_size=(x_shape[1], x_shape[2]), padding='valid', name='V5')(x)
        anchors = Flatten()(anchors)

        return anchors

    def anchorLoss(self, y_true, y_pred):
        diff = y_true - y_pred
        return tf.sqrt(tf.reduce_mean(tf.square(0.3*diff[:, :(3*self.num_anchor)])+tf.square(diff[:, (3*self.num_anchor):])))
