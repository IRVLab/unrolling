from __future__ import print_function, division
import os
import sys
import datetime
import numpy as np
from scipy import misc
# keras libs
import keras.backend as K
from keras.optimizers import Adam
# local libs
from nets.hfnet import Res_HFNet
from utils.data_utils import dataLoaderTUM

## dataset and experiment directories
dataset_name = "TUM"
# data_dir = "/mnt/data2/unRolling_Shutter/Rolling_Shutter_TUM/"
data_dir = "/home/jiawei/Workspace/unrolling/data/tum"
seq = 1 # None for all

## input/output shapes
im_shape = (256, 320)
data_loader = dataLoaderTUM(data_path=data_dir, seq_no=seq, res=(im_shape[1], im_shape[0])) 
# training parameters
num_epochs = 20
batch_size = 1
ckpt_interval = 2 # per epoch
steps_per_epoch = (data_loader.num_train//batch_size)
num_step = num_epochs*steps_per_epoch
#####################################################################

model_name = "HFNet"
model_loader = Res_HFNet(im_shape)
model_loader.model.compile(optimizer=Adam(1e-4, 0.5), loss=model_loader.flow_loss_VGG)

# checkpoint directory
checkpoint_dir = os.path.join("checkpoints/", dataset_name)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
#####################################################################


print ("\nTraining: {0} with {1} data".format(model_name, dataset_name))
## training pipeline
step, epoch = 0, 0
while (step <= num_step):
    for i, (imgs_gs, imgs_rs) in enumerate(data_loader.load_batch(batch_size)):
        # train the generator
        loss = model_loader.model.train_on_batch(imgs_rs, [imgs_gs])
        # increment step, and show the progress 
        step += 1
        if (step%100==0): print ("Loss at step {0}: {1}".format(step, loss))

    # increment epoch, save model at regular intervals 
    epoch += 1
    ## save model and weights
    if (epoch%ckpt_interval==0):
        ckpt_name = os.path.join(checkpoint_dir, ("model_%d" %epoch))
        with open(ckpt_name+"_.json", "w") as json_file:
            json_file.write(model_loader.model.to_json())
        model_loader.model.save_weights(ckpt_name+"_.h5")
        print("\nSaved trained model in {0}\n".format(checkpoint_dir))



