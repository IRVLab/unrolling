from __future__ import print_function, division
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from data_loader import dataLoader
from rectifier import rectifier

# read num_anchor from command line
parser = argparse.ArgumentParser()
parser.add_argument('--anchor', help='Number of anchors to predict')
parser.add_argument(
    '--rectify_img', help='Whether to rectify images', default=0)
args = parser.parse_args()
num_anchor = int(args.anchor)
print('Number of anchors: {}'.format(num_anchor))
rectify_img = True if int(args.rectify_img) > 0 else False
print('Rectify image: {}'.format(rectify_img))

# load data
data_loader = dataLoader()
imgs = data_loader.loadSeqImg()
flows = data_loader.loadSeqFlow()
depths = data_loader.loadSeqDepth()
anchors = data_loader.loadSeqAnchor(num_anchor)
total_count = len(imgs)

# path to save results
save_path = os.path.join(os.getcwd(), "test_results/seq2/")
if not os.path.exists(save_path):
    os.makedirs(save_path)
img_path = save_path+'images/'
if rectify_img and not os.path.exists(img_path):
    os.makedirs(img_path)

wins = 0
input_errs = []
errs = []
imgs_res = []
for i in tqdm(range(total_count)):
    img_input_rgb = np.expand_dims(imgs[i], 0)  # (1, h, w, 3)
    img_input = np.expand_dims(img_input_rgb[:, :, :, 0], -1)  # (1, h, w, 1)
    flow_gt = flows[i]

    if num_anchor > 0:
        flow = rectifier.getGS2RSFlow(depths[i], data_loader.cam, anchors[i])
    else:
        flow = flow_gt

    pred_dist = np.nanmean(np.sqrt(np.sum(np.square(flow_gt-flow), axis=-1)))
    zero_dist = np.nanmean(np.sqrt(np.sum(np.square(flow_gt), axis=-1)))
    input_errs.append(zero_dist)
    errs.append(pred_dist)
    wins += (1 if zero_dist > pred_dist else 0)

    if rectify_img:
        img_rgb = 255*img_input_rgb[0]
        img_rectified = rectifier.rectifyImgByFlow(img_rgb, flow)
        cv2.imwrite(img_path+'{}.png'.format(i), img_rectified)

print('Improved Ratio: {:.3f}'.format(wins/len(errs)))
print('Input EPE errs: {:.3f}'.format(np.mean(input_errs)))
print('EPE errs:       {:.3f}'.format(np.mean(errs)))
np.save(save_path+'errs{}.npy'.format(num_anchor), np.array(errs))
