from __future__ import print_function, division
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from data_loader import dataLoader
from model.depthnet import DepthNet

# load data
data_loader = dataLoader()
imgs = data_loader.loadTrainingImg(grayscale=True)
depths = data_loader.loadTrainingDepth()
v_imgs = data_loader.loadValidationImg(grayscale=True)
v_depths = data_loader.loadValidationDepth()

# load model
depthnet = DepthNet(data_loader.getImgShape())

# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt_name = os.path.join(checkpoint_dir, 'model_depth.hdf5')
checkpoint = ModelCheckpoint(
    ckpt_name, save_weights_only=True, save_best_only=True)

# tensorboard
tensorboard_cb = TensorBoard(log_dir='./.logs/depth')

# parameters
epochs = 200
batch_size = 100
lr = 1e-4
decay = 9 / (epochs * imgs.shape[0] / batch_size)  # decay by 0.1 at the end

# training
depthnet.model.compile(optimizer=Adam(lr=lr, decay=decay), loss=depthnet.depthLoss)
depthnet.model.fit(imgs, depths, validation_data=(v_imgs, v_depths), batch_size=batch_size,
                epochs=epochs, callbacks=[checkpoint, tensorboard_cb])
