from __future__ import absolute_import,division,print_function
import os
import shutil
import numpy as np

from stereo_rectifier import stereoRectify
from pose_handler import getPoses
from depth_estimator import getDepth
from gt_flow_generator import getGS2RSFlows

if __name__=="__main__":
    resolution = (320,256)
    ns_per_v = 29.4737*1024/resolution[1]*1000
    seqs = ['1','3','5','2','4','7','9','6','8','10']
    total_img_count = 0
    for seq in seqs:
        print ('\n\n\nProcessing Sequence '+str(seq))
        data_dir = '/home/moxxx066/Workspace/data/dataset-seq{}/dso/'.format(seq)
        save_dir = os.path.join(os.getcwd(),'../data/seq{}/'.format(seq))

        if os.path.exists(save_dir): shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        print ('Stereo Rectifying...')
        stereoRectify(data_dir, save_dir, resolution)

        print ('Getting Pose...')
        getPoses(data_dir, save_dir, resolution, ns_per_v)

        print ('Getting Depth...')
        getDepth(save_dir)

        print ('Getting Unrolling Flow...')
        getGS2RSFlows(save_dir)
        

        img0_dir = save_dir+"cam0/images/"
        img_count = len(os.listdir(img0_dir))
        total_img_count += img_count

    # randomly pick 80%/20% data for training/testing
    indices = np.random.permutation(total_img_count)
    last_train_idx = int(0.8*total_img_count)
    train_idx = indices[:last_train_idx]
    test_idx = indices[last_train_idx:]
    idx_folder = os.path.join(os.getcwd(),'../data/')
    np.save(idx_folder+'train_idx.npy', train_idx)
    np.save(idx_folder+'test_idx.npy', test_idx)