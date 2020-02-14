#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
import os
import numpy as np
import cv2
import tensorflow as tf
from nets.utils import dataLoaderTUM, draw_flow

## dataset and experiment directories
# data_dir = "/mnt/data2/unRolling_Shutter/Rolling_Shutter_TUM/"
data_dir = "/home/jiawei/Workspace/data/datasets/TUM/"
seq = 1 # None for all

## input/output shapes
im_shape = (256, 320) # should be multiples of 64 to avoid PWC padding
data_loader = dataLoaderTUM(data_path=data_dir, seq_no=seq, res=(im_shape[1], im_shape[0])) 

## PWC Net stuffs
sess = tf.Session()
with tf.gfile.GFile('nets/pwcnet.pb', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name="")
x_tnsr = sess.graph.get_tensor_by_name('x_tnsr:0') 
flow_pred = sess.graph.get_tensor_by_name('pwcnet/flow_pred:0') # pwc y flow  


def img2pwc(img):
    img_01 = img / 127.5      # [0,255] -> [0, 1]
    img_01c3 = np.zeros((img.shape[0],img.shape[1],3), dtype=np.float32)
    img_01c3[:,:,0] = img_01[:,:,0]
    img_01c3[:,:,1] = img_01[:,:,0]
    img_01c3[:,:,2] = img_01[:,:,0]
    return img_01c3

def filter_flow(img, flow):
    h,w = flow.shape[:2]
    edges = cv2.Canny(img, 40, 80)
    sum_hy = []
    p_c = 0
    n_c = 0
    for hi in range(h):
        for wi in range(w):
            if not edges[hi, wi]: 
                flow[hi,wi,:] = 0
            else:
                wf,hf = flow[hi,wi].T
                sum_hy.append(hf)
                if hf>0:
                    p_c += 1
                else:
                    n_c += 1
    print(f'{np.mean(sum_hy)} {p_c} - {n_c}')
    return flow

def main():
    flow_dir = os.path.join(data_dir, f'seq{seq}/flow')
    print(flow_dir)
    if not os.path.exists(flow_dir): os.makedirs(flow_dir)
    for i in range(data_loader.num_train):
        imgs_gs, imgs_rs, _ = data_loader.load_batch(i, batch_size=1)
        img_gs, img_rs = imgs_gs[0], imgs_rs[0]
        img_pair = np.array([[img2pwc(img_rs), img2pwc(img_gs)]], dtype=np.float32) 
        flow_rs2gs = sess.run(flow_pred, feed_dict={x_tnsr: img_pair})[0]
        filtered_flow_rs2gs = filter_flow(img_rs, flow_rs2gs)

        fname = os.path.join(flow_dir, f'{i:0>4d}')
        np.save(fname,filtered_flow_rs2gs)
        
        cv2.imshow('flow', draw_flow(img_rs, filtered_flow_rs2gs))
        cv2.waitKey(1)


if __name__ == '__main__':
    main()



