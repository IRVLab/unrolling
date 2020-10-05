from __future__ import print_function, division
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from data_loader import dataLoader
from model.unrollnet import UnrollNet

# load data
data_loader = dataLoader()
imgs = data_loader.loadTrainingImg(grayscale=True)
flows = data_loader.loadTrainingFlow()
v_imgs = data_loader.loadValidationImg(grayscale=True)
v_flows = data_loader.loadValidationFlow()

# load model
unrollnet = UnrollNet(data_loader.getImgShape())

# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt_name = os.path.join(checkpoint_dir, 'model_unroll.hdf5')
checkpoint = ModelCheckpoint(
    ckpt_name, save_weights_only=True, save_best_only=True)

# tensorboard
tensorboard_cb = TensorBoard(log_dir='./.logs/unroll')

# parameters
epochs = 200
batch_size = 10
lr = 1e-4
decay = 9 / (epochs * imgs.shape[0] / batch_size)  # decay by 0.1 at the end

unrollnet.model.compile(optimizer=Adam(
    lr=lr, decay=decay), loss=unrollnet.flowLoss)
unrollnet.model.fit(imgs, flows, validation_data=(v_imgs, v_flows), batch_size=batch_size,
                    epochs=epochs, callbacks=[checkpoint, tensorboard_cb])
