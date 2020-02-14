import tensorflow as tf
from keras.models import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.layers import add, Lambda, ZeroPadding2D, UpSampling2D, Reshape
from keras import backend as K
import numpy as np
from copy import deepcopy

# utility functions: identity block
def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    """
    filters1, filters2, filters3 = filters
    # naming
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # sub-block1
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # sub-block2
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # sub-block3
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
    # output activation
    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


# utility functions: conv block
def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    """
    filters1, filters2, filters3 = filters   
    # naming 
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    # sub-block1
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # sub-block2
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # sub-block3
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
    # shortcut convs
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)
    # output activation
    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def get_resnet_encoder(img_input):
    # sub-block1
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(32, (5, 5), strides=(2, 2), name='conv1')(x)
    f1 = x
    # sub-block2
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3) , strides=(2, 2))(x)
    x = conv_block(x, 3, [32, 32, 128], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='b')
    x = identity_block(x, 3, [32, 32, 128], stage=2, block='c')
    f2 = Lambda(lambda x : x[: , :-1 , :-1 , :  ] )(x) # one-sided padding
    # sub-block3
    x = conv_block(x, 3, [64, 64, 256], stage=3, block='a')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='c')
    x = identity_block(x, 3, [64, 64, 256], stage=3, block='d')
    f3 = x 
    # sub-block4
    x = conv_block(x, 3, [128, 128, 512], stage=4, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=4, block='d')
    f4 = x 
    # sub-block5
    x = conv_block(x, 3, [256, 256, 2048], stage=5, block='a')
    x = identity_block(x, 3, [256, 256, 2048], stage=5, block='b')
    f5 = x 
    # return
    return [f1 , f2 , f3 , f4 , f5]


def hfnet_decoder(fin):
	f1 = fin
	f1 = ZeroPadding2D((1,1))(f1)
	f1 = Conv2D(256, (3, 3), padding='valid')(f1)
	f1 = BatchNormalization()(f1)
	f1 = UpSampling2D((2,2))(f1)
	f1 = ZeroPadding2D((1,1))(f1)
	f1 = Conv2D(128, (3, 3), padding='valid')(f1)
	f1 = BatchNormalization()(f1)

	f2 = UpSampling2D((2,2))(f1)
	f2 = ZeroPadding2D((1,1))(f2)
	f2 = Conv2D(64, (3, 3), padding='valid')(f2)
	f2 = BatchNormalization()(f2)

	f3 = UpSampling2D((2,2))(f2)
	f3 = ZeroPadding2D((1,1))(f3)
	f3 = Conv2D(32, (3, 3), padding='valid')(f3)
	f3 = BatchNormalization()(f3)
	out = Conv2D(2, (3, 3), padding='same')(f3) 
        # output shape: <batch-size, 256, 320, 1>
	return out


class Res_HFNet():
    """ Proposed model
    """
    def __init__(self, im_shape):
        self.im_shape = im_shape
        self.encoder_level = 2
        self.model = self.build_hfnet()

    def build_hfnet(self):
        img_input = Input(shape=(self.im_shape[0], self.im_shape[1], 1))
        features = get_resnet_encoder(img_input)
        flow_output = hfnet_decoder(features[self.encoder_level]) 
        # HFNet will output the PWC flow, which we will find mse loss against zero matrix
        return Model(input=img_input, output=flow_output)

    def flow_loss(self, flow_rs2gs, flow_rs2pred_gs):
        # # We don't wish the net to distort along x direction
        # flow_rs2pred_gs_x_disp = Lambda(lambda x : x[:,:,:,0])(flow_rs2pred_gs)
        # loss_pred_x_disp = K.mean(K.square(flow_rs2pred_gs_x_disp))

        # We wish the final flow to have zero displacement along y direction because of the stereo config
        flow_gs2pred_gs = -flow_rs2gs + flow_rs2pred_gs
        flow_gs2pred_gs_y_disp = Lambda(lambda x : x[:,:,:,1])(flow_gs2pred_gs)
        loss_final_y_disp = K.mean(K.square(flow_gs2pred_gs_y_disp))

        return loss_final_y_disp