from __future__ import print_function, division
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from data_loader import dataLoader
from evaluation.depthnet import DepthNet

# parameters
batch_size = 100
no_epochs = 1000
learning_rate = 1e-6

# load data
data_loader = dataLoader() 
imgs, depths = data_loader.loadTrainingDepth()

# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
ckpt_name = os.path.join(checkpoint_dir, 'model_depth.hdf5')
checkpoint = ModelCheckpoint(ckpt_name, monitor='loss', save_weights_only=True, save_best_only=True)

# trainning
model_loader = DepthNet(data_loader.getImgShape())
model_loader.model.compile(optimizer=Adam(learning_rate), loss=model_loader.depthLoss)
model_loader.model.load_weights(ckpt_name)
model_loader.model.fit(imgs, depths, batch_size=batch_size, epochs=no_epochs, callbacks=[checkpoint])

