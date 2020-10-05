from __future__ import absolute_import, division, print_function
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten
from classification_models.keras import Classifiers


def Conv2d_BN_Relu(x, filters, kernel_size, name, strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding,
               strides=strides, name=name+'_conv')(x)
    x = BatchNormalization(axis=3, name=name+'_bn')(x)
    x = Activation('relu')(x)
    return x


class AnchorNet():
    def __init__(self, im_shape, num_anchor):

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

        self.model = Model(inputs=resnet34_model.input, outputs=vel)
