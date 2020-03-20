from __future__ import absolute_import,division,print_function
from numpy import linalg as LA
import numpy as np

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
    
