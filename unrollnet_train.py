from __future__ import print_function, division
import os
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from data_loader import dataLoader
from unrollnet import UnrollNet

# parameters
batch_size = 10
no_epochs = 200
learning_rate = 1e-7

# load data
data_dir = os.path.join(os.getcwd(), "data/")
seqs = [1,2,3,4,5,6,7,8,9,10] 
data_loader = dataLoader(data_path=data_dir, seqs=seqs) 
imgs_rs, flows = data_loader.loadTraining()

# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
ckpt_name = os.path.join(checkpoint_dir, 'model.hdf5')
checkpoint = ModelCheckpoint(ckpt_name, monitor='loss', save_weights_only=True, save_best_only=True, period=5)

# trainning
model_loader = UnrollNet(data_loader.getImgShape())
model_loader.model.compile(optimizer=Adam(learning_rate), loss=model_loader.flowLoss)
model_loader.model.load_weights(ckpt_name)
model_loader.model.fit(imgs_rs, flows, batch_size=batch_size, epochs=no_epochs, callbacks=[checkpoint])

