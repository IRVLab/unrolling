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

from __future__ import absolute_import, division, print_function
import os
import shutil
import numpy as np

from stereoRectifier import stereoRectify, stereoRemap
from poseHandler import getPoses
from depthEstimator import getDepth
from gtFlowGenerator import getRS2GSFlows
from gsImgRectifier import rectify_gs_imgs

data_path = '/mnt/data2/jiawei/unrolling/tum_data/'
save_path = os.getcwd()+'/../data/'
seqs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
output_resolution = (320, 256)
ns_per_v = 29.4737*1023/(output_resolution[1]-1)*1000

for seq in seqs:
    print('\n\n\nProcessing Sequence '+str(seq))
    seq_path = os.path.join(data_path, 'dataset-seq{}/dso/'.format(seq))
    seq_save_path = os.path.join(save_path, 'seq{}/'.format(seq))

    print(seq_save_path)
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
    getRS2GSFlows(seq_save_path, ns_per_v)

    print('Getting GS Image...')
    rectify_gs_imgs(seq_save_path, output_resolution)

    # clean up
    cam0_folder = os.path.join(seq_save_path, 'cam0/')
    shutil.rmtree(cam0_folder) if os.path.exists(cam0_folder) else None
    tmp_files = ['T_cam0_v1.npy', 'valid_ns.npy', 'cam1/stereo_map.npy']
    for file in tmp_files:
        file_path = os.path.join(seq_save_path, file)
        os.remove(file_path) if os.path.exists(file_path) else None
