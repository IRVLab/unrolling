# Learning Rolling Shutter Correction from Real Data without Camera Motion Assumption
# Copyright (C) <2020> <Jiawei Mo, Md Jahidul Islam, Junaed Sattar>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import print_function, division
import argparse
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from data_loader import dataLoader
from model.anchornet import AnchorNet

# read num_anchor from command line
parser = argparse.ArgumentParser()
parser.add_argument('--anchor', help='Number of anchors to predict')
args = parser.parse_args()
num_anchor = int(args.anchor)
print('Number of anchors to predict: {}'.format(num_anchor))

# load data
data_loader = dataLoader()
imgs = data_loader.loadTrainingImg()
anchors = data_loader.loadTrainingAnchor(num_anchor)
v_imgs = data_loader.loadValidationImg()
v_anchors = data_loader.loadValidationAnchor(num_anchor)

# load model
anchornet = AnchorNet(data_loader.getImgShape(), num_anchor)

# checkpoint
checkpoint_path = os.path.join(os.getcwd(), "checkpoints/")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
ckpt_name = os.path.join(
    checkpoint_path, 'model_anchor{}.hdf5'.format(num_anchor))
checkpoint_cb = ModelCheckpoint(
    ckpt_name, save_weights_only=True, save_best_only=True)

# tensorboard
tensorboard_cb = TensorBoard(log_dir='./.logs/{}'.format(num_anchor))

# parameters
epochs = 200
batch_size = 40
lr = 1e-4
decay = 9 / (epochs * imgs.shape[0] / batch_size)  # decay by 0.1 at the end

# training
anchornet.model.compile(optimizer=Adam(
    lr=lr, decay=decay), loss=anchornet.anchorLoss)
# anchornet.model.load_weights(ckpt_name)
anchornet.model.fit(imgs, anchors, validation_data=(v_imgs, v_anchors), batch_size=batch_size,
                    epochs=epochs, callbacks=[checkpoint_cb, tensorboard_cb])
