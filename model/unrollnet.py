from __future__ import absolute_import, division, print_function
from keras.models import Input, Model
from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, add, Dropout
import tensorflow as tf


def plainEncoderBlock(input_tensor, filters, strides, stage):
    x = Conv2D(filters, kernel_size=3, strides=strides, padding='same',
               name='plainEncoderBlock_'+str(stage)+'ac')(input_tensor)
    x = BatchNormalization(
        axis=-1, name='plainEncoderBlock_'+str(stage)+'ab')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=3, strides=1, padding='same',
               name='plainEncoderBlock_'+str(stage)+'bc')(x)
    x = BatchNormalization(
        axis=-1, name='plainEncoderBlock_'+str(stage)+'bb')(x)
    x = Activation('relu')(x)

    return x


def plainDecoderBlock(input_tensor, filters, strides, stage):
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same',
               name='plainDecoderBlock'+str(stage)+'_ac')(input_tensor)
    x = BatchNormalization(
        axis=-1, name='plainDecoderBlock'+str(stage)+'_ab')(x)
    x = Activation('relu')(x)

    if strides == 1:
        x = Conv2D(filters, kernel_size=3, strides=1, padding='same',
                   name='plainDecoderBlock'+str(stage)+'_bc')(x)
    else:
        x = Conv2DTranspose(filters, kernel_size=2, strides=2,
                            padding='same', name='plainDecoderBlock'+str(stage)+'_bc')(x)

    x = BatchNormalization(
        axis=-1, name='plainDecoderBlock'+str(stage)+'_bb')(x)
    x = Activation('relu')(x)

    return x


def resEncoderBlock(input_tensor, filters, strides, stage):
    x = Conv2D(filters, kernel_size=3, strides=strides, padding='same',
               name='resEncoderBlock_'+str(stage)+'ac')(input_tensor)
    x = BatchNormalization(axis=-1, name='resEncoderBlock_'+str(stage)+'ab')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, kernel_size=3, strides=1, padding='same',
               name='resEncoderBlock_'+str(stage)+'bc')(x)
    x = BatchNormalization(axis=-1, name='resEncoderBlock_'+str(stage)+'bb')(x)

    res = input_tensor
    if strides != 1:
        res = Conv2D(filters, kernel_size=1, strides=strides,
                     padding='same', name='resEncoderBlock_'+str(stage)+'rc')(res)
        res = BatchNormalization(
            axis=-1, name='resEncoderBlock_'+str(stage)+'rb')(res)
    x = add([x, res])

    x = Activation('relu')(x)

    return x


def resDecoderBlock(input_tensor, filters, strides, stage):
    x = Conv2D(filters, kernel_size=3, strides=1, padding='same',
               name='resDecoderBlock_'+str(stage)+'ac')(input_tensor)
    x = BatchNormalization(axis=-1, name='resDecoderBlock_'+str(stage)+'ab')(x)
    x = Activation('relu')(x)

    if strides == 1:
        x = Conv2D(filters, kernel_size=3, strides=1, padding='same',
                   name='resDecoderBlock_'+str(stage)+'bc')(x)
    else:
        x = Conv2DTranspose(filters, kernel_size=2, strides=2,
                            padding='same', name='resDecoderBlock_'+str(stage)+'bc')(x)
    x = BatchNormalization(axis=-1, name='resDecoderBlock_'+str(stage)+'bb')(x)

    res = input_tensor
    if strides != 1:
        res = Conv2DTranspose(filters, kernel_size=1, strides=2,
                              padding='same', name='resDecoderBlock_'+str(stage)+'rc')(res)
        res = BatchNormalization(
            axis=-1, name='resDecoderBlock_'+str(stage)+'rb')(res)
    x = add([x, res])

    x = Activation('relu')(x)

    return x


def EncoderNet(img_input):
    # input_layer
    x = Conv2D(64, kernel_size=3, strides=1,
               padding='same', name='input_c')(img_input)
    x = BatchNormalization(axis=-1, name='input_b')(x)
    x = Activation('relu')(x)
    # en_layer1
    x = plainEncoderBlock(x, filters=64, strides=1, stage=1)
    # en_layer2
    x = resEncoderBlock(x, filters=128, strides=2, stage=2)
    # en_layer3
    x = resEncoderBlock(x, filters=256, strides=2, stage=3)
    # en_layer4
    x = resEncoderBlock(x, filters=512, strides=2, stage=4)
    # en_layer5
    x = resEncoderBlock(x, filters=512, strides=2, stage=5)
    # en_layer6
    features = resEncoderBlock(x, filters=512, strides=1, stage=6)

    return features


def DecoderNet(fin):
    # de_layer6
    x = resDecoderBlock(fin, filters=512, strides=1, stage=6)
    # de_layer5
    x = resDecoderBlock(x, filters=512, strides=2, stage=5)
    # de_layer4
    x = resDecoderBlock(x, filters=256, strides=2, stage=4)
    # de_layer3
    x = resDecoderBlock(x, filters=128, strides=2, stage=3)
    # de_layer2
    x = resDecoderBlock(x, filters=64, strides=2, stage=2)
    # de_layer1
    x = plainDecoderBlock(x, filters=64, strides=1, stage=1)
    # output_layer
    out = Conv2D(2, kernel_size=3, strides=1, padding='same', name='output')(x)

    return out


class UnrollNet():
    def __init__(self, im_shape):
        img_rs = Input(shape=(im_shape[0], im_shape[1], 1))
        features = EncoderNet(img_rs)
        flow_gs2rs = DecoderNet(features)
        self.model = Model(input=img_rs, output=flow_gs2rs)

    def flowLoss(self, y_true, y_pred):
        diff = tf.where(tf.is_nan(y_true),
                        tf.zeros_like(y_true), y_true-y_pred)
        mean = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(diff), -1)))
        return mean
