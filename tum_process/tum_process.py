from __future__ import absolute_import, division, print_function
import os
import numpy as np

from stereoRectifier import stereoRectify, stereoRemap
from poseHandler import getPoses
from depthEstimator import getDepth
from gtFlowGenerator import getGS2RSFlows

data_path = '/mnt/data2/jiawei/unrolling/tum_data/'
save_path = os.path.join(os.getcwd(), '/data/')
seqs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
output_resolution = (320, 256)
ns_per_v = 29.4737*1023/(output_resolution[1]-1)*1000

for seq in seqs:
    print('\n\n\nProcessing Sequence '+str(seq))
    seq_path = os.path.join(data_path, 'dataset-seq{}/dso/'.format(seq))
    seq_save_path = os.path.join(save_path, 'seq{}/'.format(seq))

    if not os.path.exists(seq_save_path):
        os.makedirs(seq_save_path)

    stereoRectify(seq_path, seq_save_path, output_resolution)

    print('Getting Pose...')
    getPoses(seq_path, seq_save_path, output_resolution[1], ns_per_v)

    print('Stereo Rectifying...')
    stereoRemap(seq_path, seq_save_path)

    print('Getting Depth...')
    getDepth(seq_save_path)

    print('Getting Unrolling Flow...')
    getGS2RSFlows(seq_save_path, ns_per_v)

# randomly pick 80%/20% data for training/testing
total_img_count = 0
for seq in seqs:
    depth_path = os.path.join(save_path, 'seq{}/cam1/depth/'.format(seq))
    img_count = len(os.listdir(depth_path))
    total_img_count += img_count
indices = np.random.permutation(total_img_count)
last_train_idx = int(0.8*total_img_count)
last_val_idx = int(0.9*total_img_count)
train_idx = indices[:last_train_idx]
val_idx = indices[last_train_idx:last_val_idx]
test_idx = indices[last_val_idx:]
np.save(os.path.join(save_path, 'train_idx.npy'), train_idx)
np.save(os.path.join(save_path, 'val_idx.npy'), val_idx)
np.save(os.path.join(save_path, 'test_idx.npy'), test_idx)
