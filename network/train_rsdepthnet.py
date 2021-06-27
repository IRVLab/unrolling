# Learning Rolling Shutter Correction from Real Data without Camera Motion Assumption
# Copyright (C) <2021> <Jiawei Mo, Md Jahidul Islam, Junaed Sattar>

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

# fmt: off
from __future__ import print_function, division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import shutil

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from DataLoader import TumDataSet, DataGenerator
from RsDepthNet import RsDepthNet
# fmt: on

# dataset
batch_size = 32
tum = TumDataSet()
train_dg = DataGenerator(tum.data['train'], batch_size, dtype='depth')
val_dg = DataGenerator(tum.data['val'], batch_size, dtype='depth')

# weight saving directory
checkpoint_path = os.path.join(os.getcwd(), "checkpoints/")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_file = os.path.join(checkpoint_path, 'model_depth.hdf5')

# tensorboard
log_path = os.path.join(checkpoint_path, '.logs/depth')
if os.path.exists(log_path):
    shutil.rmtree(log_path)

# load model
model = RsDepthNet(tum.params)

# training
model.model.compile(optimizer=Adam(learning_rate=1e-3), loss=model.depthLoss)
model.model.fit(train_dg, validation_data=val_dg, epochs=100, callbacks=[
                ModelCheckpoint(checkpoint_file,
                                save_weights_only=True, save_best_only=True),
                TensorBoard(log_dir=log_path)])
