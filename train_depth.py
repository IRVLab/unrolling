from __future__ import print_function, division
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from data_loader import dataLoader
from model.depthnet import DepthNet

# parameters
batch_size = 100
no_epochs = 200

# load data
data_loader = dataLoader()
imgs = data_loader.loadTrainingImg()
depths = data_loader.loadTrainingDepth()

# load model
model = DepthNet(data_loader.getImgShape())

# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), "model/checkpoints")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt_name = os.path.join(checkpoint_dir, 'model_depth.hdf5')
checkpoint = ModelCheckpoint(
    ckpt_name, monitor='loss', save_weights_only=True, save_best_only=True)

# training
learning_rate = 1e-4
model.model.compile(optimizer=Adam(learning_rate), loss=model.depthLoss)
model.model.fit(imgs, depths, batch_size=batch_size,
                epochs=no_epochs, callbacks=[checkpoint])

learning_rate = 1e-5
model.model.compile(optimizer=Adam(learning_rate), loss=model.depthLoss)
model.model.load_weights(ckpt_name)
model.model.fit(imgs, depths, batch_size=batch_size,
                epochs=no_epochs, callbacks=[checkpoint])

learning_rate = 1e-6
model.model.compile(optimizer=Adam(learning_rate), loss=model.depthLoss)
model.model.load_weights(ckpt_name)
model.model.fit(imgs, depths, batch_size=batch_size,
                epochs=no_epochs, callbacks=[checkpoint])
