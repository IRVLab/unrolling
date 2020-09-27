from __future__ import absolute_import, division, print_function
from keras.models import Input, Model
from keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, add, Flatten, Concatenate, Add
from keras.applications.vgg16 import VGG16

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
    def __init__(self, base='ResNet34', im_res=(320, 240), num_anchor=1):
        # input ?x320x256x1
        self.im_res = im_res
        # get features from ResNet34 or pretrained VGG16 (?, 10, 8, 512)
        if base=='VGG16': 
            inp_shape = (im_res[0], im_res[1], 3)
            img, enc_fetures =  self.VGG16(inp_shape)
        elif base=='ResNet34': 
            img = Input(shape=(im_res[0], im_res[1], 1)) 
            enc_fetures = self.ResNet34(img)
        # propagate features
        velocity_0 = self.VelocityNet(enc_fetures, num_anchor)
        velocity_est = self.ResRefNet(velocity_0, num_anchor)   
        self.model = Model(inputs=img, outputs=velocity_est)


    def ResRefNet(self, v0, num_anchor):
        # ResRefBlock 1 
        v1 = Conv2d_BN_Relu(v0, 6*num_anchor, (3, 3), name='vref1')
        v2 = Conv2d_BN_Relu(v1, 6*num_anchor, (3, 3), name='vref2')
        v2_conc = Concatenate(name='vref2_conc')([v1, v2])
        # ResRefBlock 2 
        v3 = Conv2d_BN_Relu(v2_conc, 6*num_anchor, (3, 3), name='vref3')
        v4 = Conv2d_BN_Relu(v3, 6*num_anchor, (3, 3), name='vref4')
        v4_conc = Concatenate(name='vref4_conc')([v3, v4])
        # ResRefBlock 3 (add residual) 
        v5 = Conv2d_BN_Relu(v4_conc, 6*num_anchor, (3, 3), name='vref5')
        v_res = Add(name='v_res')([v5, v0])
        # Same as before
        x = Conv2D(6*num_anchor, kernel_size=(1, 1), name='V5_ref')(v_res)  # ?x8x10x(6*num_anchor)
        x = AveragePooling2D(pool_size=(self.im_res[0]/32, self.im_res[1]/32))(x) # ?x1x1x(6*num_anchor)   1by1 conv
        vel = Flatten()(x)   # ?x(6*num_anchor)
        return vel


    def VelocityNet(self, fx, num_anchor):
        # VelocityNet
        x = Conv2d_BN_Relu(fx, filters=512, kernel_size=(3, 3), name='V0')
        x = Conv2d_BN_Relu(x, filters=256, kernel_size=(3, 3), name='V1')
        x = Conv2d_BN_Relu(x, filters=128, kernel_size=(3, 3), name='V2')
        x = Conv2d_BN_Relu(x, filters=64, kernel_size=(3, 3), name='V3')
        x = Conv2d_BN_Relu(x, filters=32, kernel_size=(3, 3), name='V4') # ?x8x10x32
        # return # ? x 8 x 10 x ( 6 * num_anchor) fetures
        x = Conv2D(6*num_anchor, kernel_size=(1, 1), name='V5')(x)  
        return x


    def VGG16(self, inp_shape):
        vgg = VGG16(input_shape=inp_shape, include_top=False, weights='imagenet')
        vgg.trainable = True
        for layer in vgg.layers:
            layer.trainable = True
        # encoder
        pool1 = vgg.get_layer('block1_pool').output
        pool2 = vgg.get_layer('block2_pool').output
        pool3 = vgg.get_layer('block3_pool').output
        pool4 = vgg.get_layer('block4_pool').output
        pool5 = vgg.get_layer('block5_pool').output
        return vgg.input, pool5 


    def ResNet34(self, img):
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
        return x



if __name__=="__main__":
    model = AnchorNet(base='ResNet34', im_res=(320, 256))
    #print (suim_net.model.summary())
