from __future__ import print_function, division
import os
import sys
import datetime
import cv2
import numpy as np
from scipy import misc
# tf keras libs
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam
# local libs
from nets.utils import dataLoaderTUM, draw_img_by_flow_batch, draw_flow
from nets.hfnet import Res_HFNet


## dataset and experiment directories
dataset_name = "TUM"
# data_dir = "/mnt/data2/unRolling_Shutter/Rolling_Shutter_TUM/"
data_dir = "/home/jiawei/Workspace/data/datasets/TUM/"
seq = 1 # None for all

## input/output shapes
im_shape = (256, 320) # should be multiples of 64 to avoid PWC padding
data_loader = dataLoaderTUM(data_path=data_dir, seq_no=seq, res=(im_shape[1], im_shape[0]), load_flow=True) 
# training parameters
num_epochs = 20
batch_size = 1
ckpt_interval = 5 # per epoch
steps_per_epoch = (data_loader.num_train//batch_size)
num_step = num_epochs*steps_per_epoch
#####################################################################

# checkpoint directory
checkpoint_dir = os.path.join("checkpoints/", dataset_name)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
#####################################################################

model_name = "HFNet"
model_loader = Res_HFNet(im_shape)
model_loader.model.load_weights('/home/jiawei/Workspace/unrolling/checkpoints/TUM/model_20_.h5')
model_loader.model.compile(optimizer=Adam(1e-3, 0.5), loss=model_loader.flow_loss)

## training pipeline
print (f"\nTraining: {model_name} with {dataset_name} data")
step, epoch = 0, 0
n_batches = [i for i in range(data_loader.num_train//batch_size-1)]
n_batches.remove(52) # file corrupted in odin?
while (step <= num_step):
    total_loss = []
    wins = 0
    for i in n_batches:
        step += 1  
        _, imgs_rs, flows_rs2gs  = data_loader.load_batch(i, batch_size=batch_size)

        # train the HFNet here
        loss = model_loader.model.train_on_batch(imgs_rs, flows_rs2gs)
        total_loss.append(loss)

##########################VISUALIZATION_AND_DEBUGING###############################################
        pred = model_loader.model.predict(imgs_rs)
        flow_rs2gen_gs = np.zeros((batch_size, im_shape[0], im_shape[1], 2), dtype=np.float32)
        flow_rs2gen_gs[:,:,:,1] = pred[:,:,:,1]
        flow_gen_gs2gs = -flow_rs2gen_gs+flows_rs2gs
        rs2gs_yoff = np.mean(np.square(flows_rs2gs[:,:,:,1]))
        gen_rs2gs_yoff = np.mean(np.square(flow_gen_gs2gs[:,:,:,1]))
        res_str = 'Decreased' if rs2gs_yoff > gen_rs2gs_yoff else 'Increased'
        print(f'{res_str} {rs2gs_yoff:.5f}=>{gen_rs2gs_yoff:.5f}')
        wins += (1 if rs2gs_yoff > gen_rs2gs_yoff else 0)
        cv2.namedWindow('gens_gs', flags=cv2.WINDOW_NORMAL)
        # gens_gs = draw_img_by_flow_batch(imgs_rs, flow_rs2gen_gs)
        # cv2.imshow('gens_gs', draw_flow(gens_gs[0],flow_gen_gs2gs[0]))
        cv2.imshow('gens_gs', draw_flow(imgs_rs[0],flows_rs2gs[0]))
        cv2.waitKey(1)

    print (f"Wins at epoch {epoch}: {wins/data_loader.num_train}")
###################################################################################################

    print (f"Loss at epoch {epoch}: {np.mean(total_loss)}")

    # increment epoch, save model at regular intervals 
    epoch += 1
    ## save model and weights
    if (epoch%ckpt_interval==0):
        ckpt_name = os.path.join(checkpoint_dir, ("model_%d" %epoch))
        with open(ckpt_name+"_.json", "w") as json_file:
            json_file.write(model_loader.model.to_json())
        model_loader.model.save_weights(ckpt_name+"_.h5")
        print("\nSaved trained model in {0}\n".format(checkpoint_dir))



