from __future__ import absolute_import, division, print_function
import sys
from copy import deepcopy

from dataset_base import _DEFAULT_DS_TRAIN_OPTIONS
from dataset_mpisintel import MPISintelDataset
from dataset_mixer import MixedDataset
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TRAIN_OPTIONS

# TODO: You MUST set dataset_root to the correct path on your machine!
_MPISINTEL_ROOT = '/home/moxxx066/Desktop/MPI-Sintel-complete'
    
# TODO: You MUST adjust the settings below based on the number of GPU(s) used for training
# Set controller device and devices
# A one-gpu setup would be something like controller='/device:GPU:0' and gpu_devices=['/device:GPU:0']
# Here, we use a dual-GPU setup, as shown below
gpu_devices = ['/device:GPU:0', '/device:GPU:1']
controller = '/device:CPU:0'

# TODO: You MUST adjust this setting below based on the amount of memory on your GPU(s)
# Batch size
batch_size = 8

# TODO: You MUST set the batch size based on the capabilities of your GPU(s) 
#  Load train dataset
ds_opts = deepcopy(_DEFAULT_DS_TRAIN_OPTIONS)
ds_opts['in_memory'] = False                          # Too many samples to keep in memory at once, so don't preload them
ds_opts['aug_type'] = 'heavy'                         # Apply all supported augmentations
ds_opts['batch_size'] = batch_size * len(gpu_devices) # Use a multiple of 8; here, 16 for dual-GPU mode (Titan X & 1080 Ti)
ds_opts['crop_preproc'] = (256, 448)                  # Crop to a smaller input size
ds_opts['type'] = 'final'
ds = MPISintelDataset(mode='train_with_val', ds_root=_MPISINTEL_ROOT, options=ds_opts)

# Display dataset configuration
ds.print_config()

# Start from the default options
nn_opts = deepcopy(_DEFAULT_PWCNET_TRAIN_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_dir'] = './pwcnet-lg-6-2-multisteps-chairsthingsmix/'
nn_opts['batch_size'] = ds_opts['batch_size']
nn_opts['x_shape'] = [2, ds_opts['crop_preproc'][0], ds_opts['crop_preproc'][1], 3]
nn_opts['y_shape'] = [ds_opts['crop_preproc'][0], ds_opts['crop_preproc'][1], 2]
nn_opts['use_tf_data'] = True # Use tf.data reader
nn_opts['gpu_devices'] = gpu_devices
nn_opts['controller'] = controller

# Use the PWC-Net-large model in quarter-resolution mode
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2

# Set the learning rate schedule. This schedule is for a single GPU using a batch size of 8.
# Below,we adjust the schedule to the size of the batch and the number of GPUs.
nn_opts['lr_policy'] = 'multisteps'
nn_opts['lr_boundaries'] = [400000, 600000, 800000, 1000000, 1200000]
nn_opts['lr_values'] = [0.0001, 5e-05, 2.5e-05, 1.25e-05, 6.25e-06, 3.125e-06]
nn_opts['max_steps'] = 1200000

# Below, we adjust the schedule to the size of the batch and our number of GPUs (2).
nn_opts['max_steps'] = int(nn_opts['max_steps'] * 8 / ds_opts['batch_size'])
nn_opts['lr_boundaries'] = [int(boundary * 8 / ds_opts['batch_size']) for boundary in nn_opts['lr_boundaries']]

# Debugging changes
nn_opts['max_to_keep'] = 50
nn_opts['display_step'] = 1000

# Instantiate the model and display the model configuration
nn = ModelPWCNet(mode='train_with_val', options=nn_opts, dataset=ds)
nn.print_config()

# Train the model
nn.train()