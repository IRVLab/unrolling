from __future__ import print_function, division
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from data_loader import dataLoader
from evaluation.depthvelnet.depthnet import DepthNet
from evaluation.depthvelnet.velocitynet import VelocityNet

# parameters
batch_size = 100
no_epochs = 200

# load data
data_loader = dataLoader() 
imgs, depths = data_loader.loadTrainingDepth()
imgs, vels = data_loader.loadTrainingVelocity()

# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

dep_ckpt_name = os.path.join(checkpoint_dir, 'model_depth.hdf5')
dep_checkpoint = ModelCheckpoint(dep_ckpt_name, monitor='loss', save_weights_only=True, save_best_only=True)
vel_ckpt_name = os.path.join(checkpoint_dir, 'model_velocity.hdf5')
vel_checkpoint = ModelCheckpoint(vel_ckpt_name, monitor='loss', save_weights_only=True, save_best_only=True)

# load network
dep_model_loader = DepthNet(data_loader.getImgShape())
vel_model_loader = VelocityNet(data_loader.getImgShape())

# DepthNet training
learning_rate = 1e-4
dep_model_loader.model.compile(optimizer=Adam(learning_rate), loss=dep_model_loader.depthLoss)
dep_model_loader.model.fit(imgs, depths, batch_size=batch_size, epochs=no_epochs, callbacks=[dep_checkpoint])

learning_rate = 1e-5
dep_model_loader.model.compile(optimizer=Adam(learning_rate), loss=dep_model_loader.depthLoss)
dep_model_loader.model.load_weights(dep_ckpt_name)
dep_model_loader.model.fit(imgs, depths, batch_size=batch_size, epochs=no_epochs, callbacks=[dep_checkpoint])

learning_rate = 1e-6
dep_model_loader.model.compile(optimizer=Adam(learning_rate), loss=dep_model_loader.depthLoss)
dep_model_loader.model.load_weights(dep_ckpt_name)
dep_model_loader.model.fit(imgs, depths, batch_size=batch_size, epochs=no_epochs, callbacks=[dep_checkpoint])

# VelocityNet training
learning_rate = 1e-4
vel_model_loader.model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')
vel_model_loader.model.fit(imgs, vels, batch_size=batch_size, epochs=no_epochs, callbacks=[vel_checkpoint])

learning_rate = 1e-5
vel_model_loader.model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')
vel_model_loader.model.load_weights(vel_ckpt_name)
vel_model_loader.model.fit(imgs, vels, batch_size=batch_size, epochs=no_epochs, callbacks=[vel_checkpoint])

learning_rate = 1e-6
vel_model_loader.model.compile(optimizer=Adam(learning_rate), loss='mean_squared_error')
vel_model_loader.model.load_weights(vel_ckpt_name)
vel_model_loader.model.fit(imgs, vels, batch_size=batch_size, epochs=no_epochs, callbacks=[vel_checkpoint])

