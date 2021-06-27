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
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import shutil
import argparse

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard

from DataLoader import TumDataSet, DataGenerator
from RsPoseNet import RsPoseNet
# fmt: on

# whether to use IMU data
parser = argparse.ArgumentParser()
parser.add_argument('--ng', action='store_const', default=False,
                    const=True, help='whether to disable gyroscope data')
parser.add_argument('--na', action='store_const', default=False,
                    const=True, help='whether to disable accelerator data')
args = parser.parse_args()
gyro = not args.ng
acc = not args.na
suffix = ('y' if gyro else 'n') + ('y' if acc else 'n')
print('gyroscope: ' + ('y' if gyro else 'n') + '; ' +
      'accelerator: ' + ('y' if acc else 'n'))

# dataset
batch_size = 32
tum = TumDataSet(gyro, acc)
train_dg = DataGenerator(tum.data['train'], batch_size, dtype='pose')
val_dg = DataGenerator(tum.data['val'], batch_size, dtype='pose')

# weight saving directory
checkpoint_path = os.path.join(os.getcwd(), "checkpoints/")
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

# tensorboard
log_path = os.path.join(checkpoint_path, '.logs/rspose_'+suffix)
if os.path.exists(log_path):
    shutil.rmtree(log_path)

# checkpoint file
checkpoint_file = os.path.join(
    checkpoint_path, 'model_rspose_{}.hdf5'.format(suffix))

# load model
model = RsPoseNet(tum.params, gyro, acc)

# training
print('Training with pose loss')
model.model.compile(optimizer=Adam(learning_rate=1e-3),
                    loss={'pose': model.poseLoss, 'flow': model.flowLoss},
                    loss_weights={'pose': 1, 'flow': 0})  # flowLoss is included just for monitoring
model.model.fit(train_dg, validation_data=val_dg, epochs=50, callbacks=[
                ModelCheckpoint(
                    checkpoint_file, save_weights_only=True, save_best_only=True),
                TensorBoard(log_dir=log_path)])

print('Training with flow loss')
model.model.compile(optimizer=Adam(learning_rate=1e-4),
                    loss={'pose': model.poseLoss, 'flow': model.flowLoss},
                    loss_weights={'pose': 0, 'flow': 1})
model.model.load_weights(checkpoint_file)
model.model.fit(train_dg, validation_data=val_dg, epochs=50, callbacks=[
                ModelCheckpoint(
                    checkpoint_file, save_weights_only=True, save_best_only=True),
                TensorBoard(log_dir=log_path)])
