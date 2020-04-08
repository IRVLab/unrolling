from __future__ import absolute_import,division,print_function
import numpy as np
from numpy import linalg as LA
from scipy.stats import norm as norm_dist

class Seed:
    def __init__(self,uv):
        self.uv = uv
        self.z_range = np.nan
        self.a = 10
        self.b = 10
        self.mu = np.nan
        self.sigma2 = np.nan
        self.vc = 0 # view count

def interpolateDepth(depth,u,v):
    h,w = depth.shape[:2]

    u0,v0 = int(u),int(v)
    u1 = u0+1 if u0<(w-1) else u0
    v1 = v0+1 if v0<(h-1) else v0

    Is = (depth[v0,u0],depth[v1,u0],depth[v0,u1],depth[v1,u1])
    Ws = ((u1-u)*(v1-v),(u1-u)*(v-v0),(u-u0)*(v1-v),(u-u0)*(v-v0))

    sum_I = sum_W = 0.0
    for i in range(4):
        if not np.isnan(Is[i]):
            sum_I += Ws[i]*Is[i]
            sum_W += Ws[i]

    res = sum_I/sum_W if sum_W>0 else np.nan
    # print([res,Is])

    return res

def transferPoint(u0,v0,depth0,T_1_0,cam0,cam1):
    d0 = interpolateDepth(depth0,u0,v0)
    if np.isnan(d0): return [np.nan,np.nan,np.nan]
    
    X0 = getRay(cam0,(u0,v0))*d0
    X1 = np.matmul(T_1_0[0:3,0:3],X0)+np.expand_dims(T_1_0[0:3,3],-1)
    u1 = X1[0,0]/X1[2,0]*cam1[0] + cam1[2]
    v1 = X1[1,0]/X1[2,0]*cam1[1] + cam1[3]
    d1 = LA.norm(X1)
    return [u1,v1,d1]

def getRay(cam,uv):
    ray_uv = np.ones(3,dtype=np.float32)
    ray_uv[0] = (uv[0]-cam[2]) / cam[0]
    ray_uv[1] = (uv[1]-cam[3]) / cam[1]
    return np.expand_dims(ray_uv / LA.norm(ray_uv),-1)

def filterDepthByFlow(cam,seeds,flow_cur2fi,T_cur_fi):
    for i in range(len(seeds)):
        [cu,cv] = seeds[i].uv
        if not np.isnan(flow_cur2fi[cv,cu,0]):
            fu,fv = flow_cur2fi[cv,cu,:]
            seeds[i] = updateSeed(cam,cam,seeds[i],(cu+fu,cv+fv),T_cur_fi)
    return seeds

def updateSeed(cam_ref,cam_cur,seed,uv_cur,T_ref_cur):
    uv_ref = seed.uv
    z = depthFromTriangulation(cam_ref,cam_cur,T_ref_cur,uv_ref,uv_cur)
    if z>=0:
        tau = computeTau(cam_ref,T_ref_cur,uv_ref,z)
        tau_inverse = 0.5 * (1.0/max(0.0000001,z-tau) - 1.0/(z+tau))
        seed=updateSeedDist(seed,1.0/z,tau_inverse*tau_inverse)
    return seed

################################ helpers for updateSeed ###########################################
def depthFromTriangulation(cam_ref,cam_cur,T_ref_cur,uv_ref,uv_cur):
    T_cur_ref = LA.inv(T_ref_cur)
    R_cur_ref = T_cur_ref[0:3,0:3]
    t_cur_ref = T_cur_ref[0:3,3]
    ray_uv_ref = getRay(cam_ref,uv_ref)
    ray_uv_cur = getRay(cam_cur,uv_cur)

    A = np.hstack((np.matmul(R_cur_ref,ray_uv_ref),ray_uv_cur))
    AtA = np.matmul(A.T,A)
    if LA.det(AtA) < 1e-5:
        return -1
    depth2 = - np.matmul(np.matmul(LA.inv(AtA),A.T),t_cur_ref)
    depth = np.fabs(depth2[0])

    return depth

def computeTau(cam_ref,T_ref_cur,uv_ref,z):
    px_error_angle = cam_ref[4]
    ray_uv_ref = getRay(cam_ref,uv_ref)
    t = np.expand_dims(T_ref_cur[0:3,3],-1)
    a = ray_uv_ref*z-t
    t_norm = LA.norm(t)
    a_norm = LA.norm(a)
    alpha = np.arccos(np.dot(ray_uv_ref.T,t)/t_norm) # dot product
    beta = np.arccos(np.dot(a.T,-t)/(t_norm*a_norm)) # dot product
    beta_plus = beta + px_error_angle
    gamma_plus = np.pi-alpha-beta_plus # triangle angles sum to PI
    z_plus = t_norm*np.sin(beta_plus)/np.sin(gamma_plus) # law of sines
    tau = z_plus - z
    return tau[0,0] # tau is 1x1 matrix
    
def updateSeedDist(seed,x,tau2):
    if seed.vc==0:
        seed.mu = x
        seed.sigma2 = tau2
        seed.vc += 1
    else:
        a = seed.a
        b = seed.b
        mu = seed.mu
        sigma2 = seed.sigma2
        norm_scale = np.sqrt(sigma2 + tau2)
        if not np.isnan(norm_scale):
            s2 = 1.0/(1.0/sigma2 + 1.0/tau2)
            m = s2*(mu/sigma2 + x/tau2)
            C1 = a/(a+b) * norm_dist.pdf(x,mu,norm_scale)
            C2 = b/(a+b) * 1.0/seed.z_range
            normalization_constant = C1 + C2
            C1 /= normalization_constant
            C2 /= normalization_constant
            f = C1*(a+1.0)/(a+b+1.0) + C2*a/(a+b+1.0)
            e = C1*(a+1.0)*(a+2.0)/((a+b+1.0)*(a+b+2.0)) \
            + C2*a*(a+1.0)/((a+b+1.0)*(a+b+2.0))

            # update parameters
            seed.a = (e - f) / (f - e / f)
            seed.b = a * (1.0 - f) / f
            mu_new = C1*m+C2*mu
            seed.mu = mu_new
            seed.sigma2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_new*mu_new
            seed.vc += 1
    
    return seed
