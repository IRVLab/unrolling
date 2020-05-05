from __future__ import print_function, division
import os
import cv2
import numpy as np
from keras.layers import Lambda
from keras.optimizers import Adam
import tensorflow as tf
import shutil
from tqdm import tqdm

from data_loader import dataLoader
from .unrollnet import UnrollNet

def rectify_img_by_flow(img, flow):
    h, w = img.shape[:2]
    indy, indx = np.indices((h, w), dtype=np.float32)
    map_x = indx.reshape(h, w).astype(np.float32) + flow[:,:,0]
    map_y = indy.reshape(h, w).astype(np.float32) + flow[:,:,1]
    rectified_img = cv2.remap(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), map_x, map_y, cv2.INTER_LINEAR)
    return rectified_img

# dataset and experiment directories
data_loader = dataLoader() 
imgs_rs, flows = data_loader.loadTesting()

ckpt_name = os.path.join(os.getcwd(), "checkpoints/model.hdf5")
model_loader = UnrollNet(data_loader.getImgShape())
model_loader.model.load_weights(ckpt_name)

save_dir = os.path.join(os.getcwd(), "test_results/")
if os.path.exists(save_dir): shutil.rmtree(save_dir)
os.makedirs(save_dir)

## training pipeline
wins = 0
ratio_img = []
for i in tqdm(range(len(imgs_rs))):
    img_input = np.expand_dims(imgs_rs[i], 0) # (1, h, w, 1)
    flow_gt = flows[i]
    flow_pred = model_loader.model.predict(img_input)[0]
    img_input = img_input[0]

    img_gt = rectify_img_by_flow(img_input, flow_gt)
    img_pred = rectify_img_by_flow(img_input, flow_pred)
    
    zero_diff = np.nanmean(np.square(flow_gt))
    pred_diff = np.nanmean(np.square(flow_gt-flow_pred))
    text_color = (0,255,0) if zero_diff > pred_diff else (0,0,255)

    img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2RGB)
    img_input = cv2.putText(img_input, 'input: '+str(zero_diff), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2) 
    img_pred = cv2.putText(img_pred, 'pred: '+str(pred_diff), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2) 
    img_gt = cv2.putText(img_gt, 'gt', (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2) 

    concat_img = cv2.hconcat([img_input,img_pred,img_gt])
    ratio_img.append([pred_diff/zero_diff, concat_img])

    wins += (1 if zero_diff > pred_diff else 0)

print ('Wins: {}'.format(wins/len(imgs_rs)))
print ('Writing results...')

ratio_img_sorted = sorted(ratio_img)
for i in range(len(imgs_rs)):
    cv2.imwrite('{}{}.png'.format(save_dir,i), ratio_img_sorted[i][1])




