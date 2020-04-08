from __future__ import absolute_import,division,print_function
import numpy as np
import os
import cv2
from tqdm import tqdm
from numpy import linalg as LA
from depth_filter import transferPoint

from flow_extractor import getFlowBD

def getRSDepth(save_dir):
    depth1_dir = save_dir+"cam1/depth/"
    
    # Load images
    img0_dir = save_dir+"cam0/images/"
    img1_dir = save_dir+"cam1/images/"
    depth0_dir = save_dir+"cam0/depth/"
    img_count = len(os.listdir(img0_dir))
    imgs0,imgs1,deps0 = [],[],[]
    for i in range(img_count):
        img0 = cv2.imread('{}{}.png'.format(img0_dir,i))
        img1 = cv2.imread('{}{}.png'.format(img1_dir,i))
        dep0 = np.load('{}{}.npy'.format(depth0_dir,i))
        imgs0.append(img0)
        imgs1.append(img1)
        deps0.append(dep0)
    h,w = imgs0[0].shape[:2]

    # Load cameras and poses
    cam0 = np.load(save_dir+"cam0/camera.npy")
    cam1 = np.load(save_dir+"cam1/camera.npy")
    T_cam0_v1 = np.load(save_dir+"poses_v1.npy")

    # Get depth by projection
    for i in tqdm(range(img_count)):        
        dep1 = np.empty([h,w])
        dep1[:] = np.nan   
        flow10 = getFlowBD(imgs1[i],imgs0[i],'Rolling Stereo Match')
        PROJ_FLOW_DIFF_THRES = 2        # Threshold to accept depth
        for v1 in range(h):
            for u1 in range(w):
                [fu,fv] = flow10[v1,u1,:]
                if np.isnan(fu): continue
                u0,v0 = u1+fu,v1+fv
                T_v1_cam0 = LA.inv(T_cam0_v1[i,v1])
                [u1_p,v1_p,d1] = transferPoint(u0,v0,deps0[i],T_v1_cam0,cam0,cam1)
                if np.isnan(u1_p): continue
                [du,dv] = [u1_p-u1,v1_p-v1]
                if (du*du+dv*dv)<PROJ_FLOW_DIFF_THRES:
                    dep1[v1,u1] = d1

        fname = os.path.join(depth1_dir,str(i))
        np.save(fname,dep1)

if __name__=="__main__":
    seqs = ['1','3','5','2','4','7','9','6','8','10']
    for seq in seqs:
        print ('\n\n\nProcessing Sequence '+str(seq))
        save_dir = os.path.join(os.getcwd(),'../data/seq{}/'.format(seq))

        getRSDepth(save_dir)