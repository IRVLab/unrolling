from __future__ import print_function, division
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from data_loader import dataLoader
from unrollnet.unrollnet import UnrollNet

# parameters
batch_size = 10
no_epochs = 200

# load data
data_loader = dataLoader() 
imgs, flows = data_loader.loadTrainingUnroll()

# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
ckpt_name = os.path.join(checkpoint_dir, 'model_unroll.hdf5')
checkpoint = ModelCheckpoint(ckpt_name, monitor='loss', save_weights_only=True, save_best_only=True)

# trainning
model_loader = UnrollNet(data_loader.getImgShape())

learning_rate = 1e-4
model_loader.model.compile(optimizer=Adam(learning_rate), loss=model_loader.flowLoss)
# model_loader.model.load_weights(ckpt_name)
model_loader.model.fit(imgs, flows, batch_size=batch_size, epochs=no_epochs, callbacks=[checkpoint])

learning_rate = 1e-5
model_loader.model.compile(optimizer=Adam(learning_rate), loss=model_loader.flowLoss)
model_loader.model.load_weights(ckpt_name)
model_loader.model.fit(imgs, flows, batch_size=batch_size, epochs=no_epochs, callbacks=[checkpoint])

learning_rate = 1e-6
model_loader.model.compile(optimizer=Adam(learning_rate), loss=model_loader.flowLoss)
model_loader.model.load_weights(ckpt_name)
model_loader.model.fit(imgs, flows, batch_size=batch_size, epochs=no_epochs, callbacks=[checkpoint])

