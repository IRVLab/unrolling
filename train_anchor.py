from __future__ import print_function, division
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from data_loader import dataLoader
from model.anchornet import AnchorNet

# read num_anchor from command line
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--num_anchor', help='Number of anchors to predict')
args = parser.parse_args()
num_anchor = int(args.num_anchor)
print('Number of anchors to predict: {}'.format(num_anchor))

# parameters
batch_size = 50
no_epochs = 300
rot_weight = 10

# load data
data_loader = dataLoader()
imgs = data_loader.loadTrainingImg()
anchors = data_loader.loadTrainingAnchor(num_anchor, rot_weight)

# load model
model_loader = AnchorNet(data_loader.getImgShape(), num_anchor)

# checkpoint
checkpoint_dir = os.path.join(
    os.getcwd(), "model/checkpoints/{}/".format(rot_weight))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
ckpt_name = os.path.join(
    checkpoint_dir, 'model_anchor{}.hdf5'.format(num_anchor))
checkpoint = ModelCheckpoint(
    ckpt_name, monitor='loss', save_weights_only=True, save_best_only=True)

# training
learning_rate = 1e-4
model_loader.model.compile(optimizer=Adam(
    learning_rate), loss='mean_squared_error')
model_loader.model.fit(imgs, anchors, batch_size=batch_size,
                       epochs=no_epochs, callbacks=[checkpoint])

learning_rate = 1e-5
model_loader.model.compile(optimizer=Adam(
    learning_rate), loss='mean_squared_error')
model_loader.model.load_weights(ckpt_name)
model_loader.model.fit(imgs, anchors, batch_size=batch_size,
                       epochs=no_epochs, callbacks=[checkpoint])

learning_rate = 1e-6
model_loader.model.compile(optimizer=Adam(
    learning_rate), loss='mean_squared_error')
model_loader.model.load_weights(ckpt_name)
model_loader.model.fit(imgs, anchors, batch_size=batch_size,
                       epochs=no_epochs, callbacks=[checkpoint])
