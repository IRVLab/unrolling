from __future__ import absolute_import, division, print_function
from keras.models import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, add, Dense, Flatten


def Conv2d_BN_Relu(x, filters, kernel_size, name, strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding,
               strides=strides, name=name+'_conv')(x)
    x = BatchNormalization(axis=3, name=name+'_bn')(x)
    x = Activation('relu')(x)
    return x


def Conv2d_BN(x, filters, kernel_size, name, strides=(1, 1), padding='same'):
    x = Conv2D(filters, kernel_size, padding=padding,
               strides=strides, activation='relu', name=name+'_conv')(x)
    x = BatchNormalization(axis=3, name=name+'_bn')(x)
    return x


def Conv_Block(inpt, filters, kernel_size, name, strides=(1, 1), shortcut=False):
    x = Conv2d_BN(inpt, filters=filters, kernel_size=kernel_size,
                  name=name+'a', strides=strides, padding='same')
    x = Conv2d_BN(x, filters=filters, kernel_size=kernel_size,
                  name=name+'b', padding='same')
    if shortcut:
        shortcut = Conv2d_BN(
            inpt, filters=filters, kernel_size=kernel_size, name=name+'s', strides=strides)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


class AnchorNet():
    def __init__(self, im_shape, num_anchor):
        img = Input(shape=(im_shape[0], im_shape[1], 1)) # ?x320x256x1
        x = Conv2d_BN(img, filters=64, kernel_size=(7, 7),
                      name='00', strides=(2, 2), padding='same')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # ResNet34: 64
        x = Conv_Block(x, filters=64, kernel_size=(3, 3), name='10')
        x = Conv_Block(x, filters=64, kernel_size=(3, 3), name='11')
        x = Conv_Block(x, filters=64, kernel_size=(3, 3), name='12')
        # ResNet34: 128
        x = Conv_Block(x, filters=128, kernel_size=(3, 3),
                       name='20', strides=(2, 2), shortcut=True)
        x = Conv_Block(x, filters=128, kernel_size=(3, 3), name='21')
        x = Conv_Block(x, filters=128, kernel_size=(3, 3), name='22')
        x = Conv_Block(x, filters=128, kernel_size=(3, 3), name='23')
        # ResNet34: 256
        x = Conv_Block(x, filters=256, kernel_size=(3, 3),
                       name='30', strides=(2, 2), shortcut=True)
        x = Conv_Block(x, filters=256, kernel_size=(3, 3), name='31')
        x = Conv_Block(x, filters=256, kernel_size=(3, 3), name='32')
        x = Conv_Block(x, filters=256, kernel_size=(3, 3), name='33')
        x = Conv_Block(x, filters=256, kernel_size=(3, 3), name='34')
        x = Conv_Block(x, filters=256, kernel_size=(3, 3), name='35')
        # ResNet34: 512
        x = Conv_Block(x, filters=512, kernel_size=(3, 3),
                       name='40', strides=(2, 2), shortcut=True)
        x = Conv_Block(x, filters=512, kernel_size=(3, 3), name='41')
        x = Conv_Block(x, filters=512, kernel_size=(3, 3), name='42')

        # VelocityNet
        x = Conv2d_BN_Relu(x, filters=512, kernel_size=(3, 3), name='V0')
        x = Conv2d_BN_Relu(x, filters=256, kernel_size=(3, 3), name='V1')
        x = Conv2d_BN_Relu(x, filters=128, kernel_size=(3, 3), name='V2')
        x = Conv2d_BN_Relu(x, filters=64, kernel_size=(3, 3), name='V3')
        x = Conv2d_BN_Relu(x, filters=32, kernel_size=(3, 3), name='V4') # ?x8x10x32

        x = Conv2D(6*num_anchor, kernel_size=(1, 1), name='V5')(x)  # ?x8x10x(6*num_anchor)
        x = AveragePooling2D(pool_size=(im_shape[0]/32, im_shape[1]/32))(x) # ?x1x1x(6*num_anchor)   1by1 conv
        vel = Flatten()(x)   # ?x(6*num_anchor)

        self.model = Model(inputs=img, outputs=vel)
