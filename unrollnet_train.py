from __future__ import print_function, division
import os
import cv2
import numpy as np
from keras.optimizers import Adam
from data_loader import dataLoader
from unrollnet import UnrollNet

## parameters
data_dir = os.path.join(os.getcwd(), "data")
seq = 1 # None for all
data_loader = dataLoader(data_path=data_dir, seq_no=seq) 
num_epochs = 500
batch_size = 4
ckpt_interval = 5 
n_batches = (data_loader.num_train//batch_size)

# checkpoint directory
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
ckpt_name = os.path.join(checkpoint_dir, ("model"))

model_loader = UnrollNet(data_loader.getImgShape())
model_loader.model.compile(optimizer=Adam(1e-4), loss=model_loader.flowLoss)
# model_loader.model.load_weights(os.path.join(checkpoint_dir, ("model.h5")))
#####################################################################

## training pipeline
step, epoch = 0, 0
for epoch in range(num_epochs):
    total_loss = []
    for i in range(n_batches):
        step += 1  
        imgs_rs, flows_gs2rs = data_loader.loadBatch(i, batch_size=batch_size)

        model_loader.model.train_on_batch(imgs_rs, flows_gs2rs)
        loss = model_loader.model.test_on_batch(imgs_rs, flows_gs2rs)
        total_loss.append(loss)

    print (f"Loss at epoch {epoch+1} / {num_epochs}: {np.mean(total_loss)}")
    ## save model and weights
    if ((epoch+1)%ckpt_interval==0):
        with open(ckpt_name+".json", "w") as json_file:
            json_file.write(model_loader.model.to_json())
        model_loader.model.save_weights(ckpt_name+".h5")
        print(f"Saved trained model in {checkpoint_dir}\n")



