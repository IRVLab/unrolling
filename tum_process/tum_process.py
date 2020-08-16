from __future__ import absolute_import, division, print_function
import os
import numpy as np

# from stereoRectifier import stereoRectify, stereoRemap
from poseHandler import getPoses
# from depthEstimator import getDepth
# from gtFlowGenerator import getGS2RSFlows

if __name__ == "__main__":
    resolution = (320, 256)
    ns_per_v = 29.4737*1023/(resolution[1]-1)*1000
    seqs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    for seq in seqs:
        print('\n\n\nProcessing Sequence '+str(seq))
        data_dir = '/home/moxxx066/Workspace/data/dataset-seq{}/dso/'.format(
            seq)
        save_dir = os.path.join(os.getcwd(), '../data/seq{}/'.format(seq))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # stereoRectify(data_dir, save_dir, resolution)

        print('Getting Pose...')
        getPoses(data_dir, save_dir, resolution[1], ns_per_v)

        # print ('Stereo Rectifying...')
        # stereoRemap(data_dir, save_dir)

        # print ('Getting Depth...')
        # getDepth(save_dir)

        # print ('Getting Unrolling Flow...')
        # getGS2RSFlows(save_dir,ns_per_v)
