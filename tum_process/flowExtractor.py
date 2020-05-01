from __future__ import absolute_import,division,print_function
import numpy as np
import cv2
from numpy import linalg as LA

from pwcnet import ModelPWCNet,_DEFAULT_PWCNET_TEST_OPTIONS

pwc_net = ModelPWCNet()

def drawFlow(img,flow):
    h,w = img.shape[:2]
    flow_vis = np.copy(img)
    edge = cv2.Canny(img,40,80)
    for v in range(h):
        for u in range(w):
            if edge[v,u]: 
                fu,fv = flow[v,u,:]
                if(fu*fu+fv*fv>1):
                    cv2.line(flow_vis,(u,v),(int(u+fu+0.5),int(v+fv+0.5)),(0,255,0))
                if(0<fu*fu+fv*fv<1):
                    cv2.circle(flow_vis,(u,v),1,(0,255,0))

    return flow_vis

def filterFlowBD(flow01,flow10):
    FLOW_THRES = 2               # threshold to accept a flow by bi-directional matching
    h,w = flow01.shape[:2]
    flow01_filtered = np.empty_like(flow01)
    flow01_filtered[:] = np.nan
    for v0 in range(h):
        for u0 in range(w):
            fu01,fv01 = flow01[v0,u0,:]
            u1,v1 = u0+fu01,v0+fv01
            u1i,v1i = int(u1+0.5),int(v1+0.5)
            if 0<=v1i<h and 0<=u1i<w:
                fu10,fv10 = flow10[v1i,u1i,:]
                du,dv = u1+fu10-u0,v1+fv10-v0
                if (du*du+dv*dv)<FLOW_THRES:
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
    
def getEH(cam0,T01,cam1):
    degenerated = LA.norm(T01[0:3,3])<1e-2
    if not degenerated:
        K0,K1,tx = np.identity((3)),np.identity(3),np.zeros((3,3))
        K0[0,0],K0[1,1],K0[0,2],K0[1,2] = cam0[0],cam0[1],cam0[2],cam0[3]
        K1[0,0],K1[1,1],K1[0,2],K1[1,2] = cam1[0],cam1[1],cam1[2],cam1[3]
        tx[2,1],tx[2,0],tx[1,0] = T01[0,3],-T01[1,3],T01[2,3]
        tx[1,2],tx[0,2],tx[0,1] = -T01[0,3],T01[1,3],-T01[2,3]
        R = T01[0:3,0:3]
        EH = np.matmul(np.matmul(np.matmul(LA.inv(K0).T,tx),R),LA.inv(K1))
    else:
        K0,K1 = np.identity((3)),np.identity(3)
        K0[0,0],K0[1,1],K0[0,2],K0[1,2] = cam0[0],cam0[1],cam0[2],cam0[3]
        K1[0,0],K1[1,1],K1[0,2],K1[1,2] = cam1[0],cam1[1],cam1[2],cam1[3]
        R = T01[0:3,0:3]
        EH = np.matmul(np.matmul(K0,R),LA.inv(K1))

    return (EH,degenerated)

def getEpipolarLine(u0,v0,EH,degenerated=False):
    if not degenerated:
        a = u0*EH[0,0]+v0*EH[1,0]+EH[2,0]
        b = u0*EH[0,1]+v0*EH[1,1]+EH[2,1]
        c = u0*EH[0,2]+v0*EH[1,2]+EH[2,2]
    else:
        if u0==0 and v0==0: # [0,0,1]*[1,1,0]^T=0
            a = EH[0,0]+EH[1,0]
            b = EH[0,1]+EH[1,1]
            c = EH[0,2]+EH[1,2]
        else:               # [u0,v0,1]*[-v0,u0,0]^T=0
            a = -v0*EH[0,0]+u0*EH[1,0]
            b = -v0*EH[0,1]+u0*EH[1,1]
            c = -v0*EH[0,2]+u0*EH[1,2]
  
    norm2 = a*a+b*b
    abc = [a,b,c]/np.sqrt(norm2)
    assert(not np.isnan(abc[0]))
    return abc

def filterFlowE(flow01,E,degenerated):
    FLOW_THRES = 1               # threshold of distance to epipolar line
    h,w = flow01.shape[:2]
    flow01_filtered = np.empty_like(flow01)
    flow01_filtered[:] = np.nan
    for v0 in range(h):
        for u0 in range(w):
            fu01,fv01 = flow01[v0,u0,:]
            u1,v1 = u0+fu01,v0+fv01
            [a,b,c] = getEpipolarLine(u0,v0,E,degenerated)
            dist = a*u1+b*v1+c
            dist = dist*dist
            if dist<FLOW_THRES:
                flow01_filtered[v0,u0,0] = flow01[v0,u0,0]
                flow01_filtered[v0,u0,1] = flow01[v0,u0,1]
    return flow01_filtered

def getFlowEp(img0,img1,cam0,T01,cam1,windowName=''):
    img_pairs = [(img0,img1),(img1,img0)]
    flow01,flow10 = pwc_net.predict_from_img_pairs(img_pairs,batch_size=2)
    flow01_filtered_bd = filterFlowBD(flow01,flow10)
    [E,degenerated] = getEH(cam0,T01,cam1)
    flow01_filtered = filterFlowE(flow01_filtered_bd,E,degenerated)

    # cv2.imshow(windowName,drawFlow(img0,flow01_filtered))
    # cv2.waitKey(1)  

    return flow01_filtered