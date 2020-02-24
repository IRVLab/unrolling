from __future__ import print_function, division
import os
import cv2
import numpy as np
# keras libs
from keras.optimizers import Adam
# local libs
from utils import dataLoaderTUM, draw_flow
from hfnet import Res_HFNet


## dataset and experiment directories
dataset_name = "TUM"
# data_dir = "/mnt/data2/unRolling_Shutter/Rolling_Shutter_TUM/"
data_dir = "/home/jiawei/Workspace/data/datasets/TUM/"
seq = 1 # None for all

## parameters
im_shape = (256, 320) # should be multiples of 64 to avoid PWC padding
data_loader = dataLoaderTUM(data_path=data_dir, seq_no=seq, out_res=(im_shape[1], im_shape[0]), load_flow=True) 
num_epochs = 500
batch_size = 4
ckpt_interval = 5 
n_batches = (data_loader.num_train//batch_size)

model_loader = Res_HFNet(im_shape)
model_loader.model.load_weights('/home/jiawei/Workspace/unrolling/checkpoints/TUM/model.h5')
model_loader.model.compile(optimizer=Adam(1e-2, 0.5), loss='mse')

# checkpoint directory
checkpoint_dir = os.path.join("checkpoints/", dataset_name)
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
ckpt_name = os.path.join(checkpoint_dir, ("model"))
#####################################################################

## training pipeline
print (f"\nTraining: HFNet with {dataset_name} data")
step, epoch = 0, 0
for epoch in range(num_epochs):
    total_loss = []
    for i in range(n_batches):
        step += 1  
        _, imgs_rs, flows_rs2gen, _  = data_loader.load_batch(i, batch_size=batch_size)

        # train the HFNet here
        loss = model_loader.model.train_on_batch(imgs_rs, flows_rs2gen)
        total_loss.append(loss)

    print (f"Loss at epoch {epoch} / {num_epochs}: {np.mean(total_loss)}")
    ## save model and weights
    if (epoch%ckpt_interval==0):
        with open(ckpt_name+".json", "w") as json_file:
            json_file.write(model_loader.model.to_json())
        model_loader.model.save_weights(ckpt_name+".h5")
        print(f"Saved trained model in {checkpoint_dir}\n")



