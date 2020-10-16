from __future__ import print_function, division
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from data_loader import dataLoader
from rectifier import rectifier

from model.depthnet import DepthNet
from model.anchornet import AnchorNet

# read num_anchor from command line
parser = argparse.ArgumentParser()
parser.add_argument('--num_anchor', help='Number of anchors to predict')
parser.add_argument(
    '--rectify_img', help='Whether to rectify images', default=0)
args = parser.parse_args()
num_anchor = int(args.num_anchor)
print('Number of anchors to predict: {}'.format(num_anchor))
rectify_img = True if int(args.rectify_img) > 0 else False
print('Rectify image: {}'.format(rectify_img))

# load data
data_loader = dataLoader()
imgs = data_loader.loadTestingImg()
flows = data_loader.loadTestingFlow()
total_count = len(imgs)

# load model
if num_anchor > 0:
    depthnet = DepthNet(data_loader.getImgShape())
    depthnet.model.load_weights(os.path.join(
        os.getcwd(), "checkpoints/model_depth.hdf5"))
    anchornet = AnchorNet(data_loader.getImgShape(), num_anchor)
    anchornet.model.load_weights(os.path.join(os.getcwd(
    ), 'checkpoints/model_anchor{}.hdf5'.format(num_anchor)))

# path to save results
save_path = os.path.join(os.getcwd(), "test_results/test/")
if not os.path.exists(save_path):
    os.makedirs(save_path)
img_path = save_path+'images/'
if rectify_img and not os.path.exists(img_path):
    os.makedirs(img_path)

wins = 0
input_errs = []
errs = []
imgs_res = []
rs_intense = []
for i in tqdm(range(total_count)):
    img_input_rgb = np.expand_dims(imgs[i], 0)  # (1, h, w, 3)
    img_input = np.expand_dims(img_input_rgb[:, :, :, 0], -1)  # (1, h, w, 1)
    flow_gt = flows[i]

    if num_anchor > 0:
        depth_pred = depthnet.model.predict(img_input)[0]
        anchor_pred = anchornet.model.predict(img_input_rgb)[0]
        flow_pred = rectifier.getGS2RSFlow(
            depth_pred, data_loader.cam, anchor_pred)
    else:
        flow_pred = np.zeros_like(flow_gt)

    pred_dist = np.nanmean(
        np.sqrt(np.sum(np.square(flow_gt-flow_pred), axis=-1)))
    zero_dist = np.nanmean(np.sqrt(np.sum(np.square(flow_gt), axis=-1)))
    input_errs.append(zero_dist)
    errs.append(pred_dist)
    wins += (1 if zero_dist > pred_dist else 0)

    if rectify_img:
        img_rgb = 255*img_input_rgb[0]
        img_rectified = rectifier.rectifyImgByFlow(img_rgb, flow_pred)
        img_rectified_gt = rectifier.rectifyImgByFlow(img_rgb, flow_gt)
        img_res = cv2.hconcat([img_rgb, img_rectified, img_rectified_gt])
        imgs_res.append(img_res)
        rs_intense.append(zero_dist)

print('Improved Ratio: {:.3f}'.format(wins/len(errs)))
print('Input EPE errs: {:.3f}'.format(np.mean(input_errs)))
print('EPE errs:       {:.3f}'.format(np.mean(errs)))
np.save(save_path+'errs{}.npy'.format(num_anchor), np.array(errs))

if rectify_img:
    # intensive rs image goes first
    rs_sorted_idx = np.argsort(rs_intense)
    for i in range(total_count):
        cv2.imwrite(img_path+'{}.png'.format(i),
                    imgs_res[rs_sorted_idx[total_count-1-i]])
