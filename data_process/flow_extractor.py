from __future__ import absolute_import,division,print_function
import numpy as np
import cv2
from numpy import linalg as LA
from numba import jit

from geometry import getEH,getEpipolarLine
from pwcnet import ModelPWCNet,_DEFAULT_PWCNET_TEST_OPTIONS

pwc_net = ModelPWCNet()

def drawFlow(img,flow):
    h,w = img.shape[:2]
    flow_vis = img
    edge = cv2.Canny(img,40,80)
    for v in range(h):
        for u in range(w):
            if edge[v,u]: 
                fu,fv = flow[v,u,:]
                if(fu*fu+fv*fv>1):
                    cv2.line(flow_vis,(u,v),(int(u+fu+0.5),int(v+fv+0.5)),(0,255,0))

    return flow_vis

@jit
def filterFlowBD(flow01,flow10):
    FLOW_THRES = 2               # threshold to accept a flow by bi-directional matching
    h,w = flow01.shape[:2]
    flow01_filtered = np.empty_like(flow01)
    flow01_filtered[:] = np.nan
    for v0 in range(h):
        for u0 in range(w):
            fu01,fv01 = flow01[v0,u0,:]
            u1,v1 = int(u0+fu01+0.5),int(v0+fv01+0.5)
            if 0<=v1<h and 0<=u1<w:
                fu10,fv10 = flow10[v1,u1,:]
                du,dv = u1+fu10-u0,v1+fv10-v0
                if (du*du+dv*dv)<FLOW_THRES:
                    flow01_filtered[v0,u0,0] = flow01[v0,u0,0]
                    flow01_filtered[v0,u0,1] = flow01[v0,u0,1]
    return flow01_filtered

@jit
def filterFlowE(flow01,E):
    FLOW_THRES = 1               # threshold of distance to epipolar line
    h,w = flow01.shape[:2]
    flow01_filtered = np.empty_like(flow01)
    flow01_filtered[:] = np.nan
    for v0 in range(h):
        for u0 in range(w):
            fu01,fv01 = flow01[v0,u0,:]
            u1,v1 = u0+fu01,v0+fv01
            [a,b,c] = getEpipolarLine(u0,v0,E)
            dist = a*u1+b*v1+c
            dist = dist*dist
            if dist<FLOW_THRES:
                flow01_filtered[v0,u0,0] = flow01[v0,u0,0]
                flow01_filtered[v0,u0,1] = flow01[v0,u0,1]
    return flow01_filtered

def getFlowBD(img0,img1,windowName=''):
    img_pairs = [(img0,img1),(img1,img0)]
    flow01,flow10 = pwc_net.predict_from_img_pairs(img_pairs,batch_size=2)
    flow01_filtered = filterFlowBD(flow01,flow10)

    # cv2.imshow(windowName,drawFlow(img0,flow01_filtered))
    # cv2.waitKey(1)  
    
    return flow01_filtered

def getFlowEp(img0,img1,cam0,T01,cam1,windowName=''):
    assert(LA.norm(T01[0:3,3])>1e-2)  # make sure Essential matrix is not degenerated

    img_pairs = [(img0,img1),(img1,img0)]
    flow01,flow10 = pwc_net.predict_from_img_pairs(img_pairs,batch_size=2)
    flow01_filtered_bd = filterFlowBD(flow01,flow10)
    [E,_] = getEH(cam0,T01,cam1)
    flow01_filtered = filterFlowE(flow01_filtered_bd,E)

    # a=np.sum(~np.isnan(flow01_filtered_bd))
    # b=np.sum(~np.isnan(flow01_filtered))
    # print([a,b])
    
    # cv2.imshow(windowName,drawFlow(img0,flow01_filtered))
    # cv2.waitKey(1)  

    return flow01_filtered