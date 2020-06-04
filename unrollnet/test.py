from __future__ import print_function, division
import os
import cv2
import numpy as np
from tqdm import tqdm

from unrollnet.unrollnet import UnrollNet

def rectifyImgByFlow(img, flow):
    h, w = img.shape[:2]
    indy, indx = np.indices((h, w), dtype=np.float32)
    map_x = indx.reshape(h, w).astype(np.float32) + flow[:,:,0]
    map_y = indy.reshape(h, w).astype(np.float32) + flow[:,:,1]
    rectified_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return rectified_img

data_folder = os.path.join(os.getcwd(), "data/seq2")
img_folder = os.path.join(data_folder, "cam1/images/")
flow_folder = os.path.join(data_folder, "cam1/flows_gs2rs/")

unrollnet = UnrollNet([256, 320])
unrollnet.model.load_weights(os.path.join(os.getcwd(), "checkpoints/model_unroll.hdf5"))

save_dir = os.path.join(os.getcwd(), "test_results/")
if not os.path.exists(save_dir+'seq2/'): os.makedirs(save_dir+'seq2/')

img_count = os.listdir(img_folder)
wins = 0
errs = []
for i in range(len(img_count)):
    img = cv2.imread(os.path.join(img_folder, str(i)+'.png'), 0)
    flow_gt = np.load(os.path.join(flow_folder, str(i)+'.npy'))

    img_input = np.expand_dims(img, 0) # (1, h, w)
    img_input = np.expand_dims(img_input, -1) # (1, h, w, 1)
    flow_pred = unrollnet.model.predict(img_input)[0]
    rectified_img = rectifyImgByFlow(img, flow_pred)

    pred_dist = np.nanmean(np.sqrt(np.sum(np.square(flow_gt-flow_pred), axis=-1)))
    zero_dist = np.nanmean(np.sqrt(np.sum(np.square(flow_gt), axis=-1)))
    wins += (1 if zero_dist > pred_dist else 0)

    cv2.imwrite('{}seq2/{}.png'.format(save_dir,i), rectified_img)
    errs.append(pred_dist)


print ('Seq2: {}'.format(wins/len(img_count)))
print ('Seq2 err: {}'.format(np.mean(errs)))

