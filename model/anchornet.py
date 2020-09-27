from __future__ import absolute_import, division, print_function
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, add, Dense, Flatten
from keras.losses import MSE
from classification_models.keras import Classifiers


def Conv2d_BN_Relu(x, filters, kernel_size, name, strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding,
               strides=strides, name=name+'_conv')(x)
    x = BatchNormalization(axis=3, name=name+'_bn')(x)
    x = Activation('relu')(x)
    return x


# def Conv2d_BN(x, filters, kernel_size, name, strides=(1, 1), padding='same'):
#     x = Conv2D(filters, kernel_size, padding=padding,
#                strides=strides, activation='relu', name=name+'_conv')(x)
#     x = BatchNormalization(axis=3, name=name+'_bn')(x)
#     return x


# def Conv_Block(inpt, filters, kernel_size, name, strides=(1, 1), shortcut=False):
#     x = Conv2d_BN(inpt, filters=filters, kernel_size=kernel_size,
#                   name=name+'a', strides=strides, padding='same')
#     x = Conv2d_BN(x, filters=filters, kernel_size=kernel_size,
#                   name=name+'b', padding='same')
#     if shortcut:
#         shortcut = Conv2d_BN(
#             inpt, filters=filters, kernel_size=kernel_size, name=name+'s', strides=strides)
#         x = add([x, shortcut])
#         return x
#     else:
#         x = add([x, inpt])
#         return x


class AnchorNet():
    def __init__(self, im_shape, num_anchor):
        # img = Input(shape=(im_shape[0], im_shape[1], 1))
        # x = Conv2d_BN(img, filters=64, kernel_size=(7, 7),
        #               name='00', strides=(2, 2), padding='same')
        # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # # ResNet34: 64
        # x = Conv_Block(x, filters=64, kernel_size=(3, 3), name='10')
        # x = Conv_Block(x, filters=64, kernel_size=(3, 3), name='11')
        # x = Conv_Block(x, filters=64, kernel_size=(3, 3), name='12')
        # # ResNet34: 128
        # x = Conv_Block(x, filters=128, kernel_size=(3, 3),
        #                name='20', strides=(2, 2), shortcut=True)
        # x = Conv_Block(x, filters=128, kernel_size=(3, 3), name='21')
        # x = Conv_Block(x, filters=128, kernel_size=(3, 3), name='22')
        # x = Conv_Block(x, filters=128, kernel_size=(3, 3), name='23')
        # # ResNet34: 256
        # x = Conv_Block(x, filters=256, kernel_size=(3, 3),
        #                name='30', strides=(2, 2), shortcut=True)
        # x = Conv_Block(x, filters=256, kernel_size=(3, 3), name='31')
        # x = Conv_Block(x, filters=256, kernel_size=(3, 3), name='32')
        # x = Conv_Block(x, filters=256, kernel_size=(3, 3), name='33')
        # x = Conv_Block(x, filters=256, kernel_size=(3, 3), name='34')
        # x = Conv_Block(x, filters=256, kernel_size=(3, 3), name='35')
        # # ResNet34: 512
        # x = Conv_Block(x, filters=512, kernel_size=(3, 3),
        #                name='40', strides=(2, 2), shortcut=True)
        # x = Conv_Block(x, filters=512, kernel_size=(3, 3), name='41')
        # x = Conv_Block(x, filters=512, kernel_size=(3, 3), name='42')

        ResNet34, preprocess_input = Classifiers.get('resnet34')
        resnet34_model = ResNet34(input_shape=(
            im_shape[0], im_shape[1], 3), weights='imagenet', include_top=False)

        # VelocityNet
        x = Conv2d_BN_Relu(resnet34_model.output, filters=512,
                           kernel_size=(3, 3), name='V0')
        x = Conv2d_BN_Relu(x, filters=256, kernel_size=(3, 3), name='V1')
        x = Conv2d_BN_Relu(x, filters=128, kernel_size=(3, 3), name='V2')
        x = Conv2d_BN_Relu(x, filters=64, kernel_size=(3, 3), name='V3')
        x = Conv2d_BN_Relu(x, filters=32, kernel_size=(3, 3), name='V4')

        x = Conv2D(6*num_anchor, kernel_size=(int(im_shape[0]/32), int(im_shape[1]/32)), padding='valid', name='V5')(x)
        vel = Flatten()(x)
        # vel = GlobalAveragePooling2D()(x)

        self.model = Model(inputs=resnet34_model.input, outputs=vel)
        # self.num_anchor = num_anchor

    # def anchor_loss(self, y_true, y_pred):
    #     # Loss for each incremental step
    #     inc_loss = MSE(y_true, y_pred)

    #     # Loss for overall velocity
    #     y_true_reshaped = tf.reshape(y_true, [-1, self.num_anchor, 6])
    #     y_pred_reshaped = tf.reshape(y_pred, [-1, self.num_anchor, 6])
    #     sum_true = tf.reduce_sum(y_true_reshaped, axis=1)
    #     sum_pred = tf.reduce_sum(y_pred_reshaped, axis=1)
    #     sum_loss = MSE(sum_true, sum_pred)

    #     return inc_loss
