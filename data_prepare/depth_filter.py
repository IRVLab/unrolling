import numpy as np
from numpy import linalg as LA
from scipy.stats import norm

def castRay(cam, ray):
    ru = ray[0,0] / ray[2,0]
    rv = ray[1,0] / ray[2,0]
    u = ru*cam[0] + cam[2]
    v = rv*cam[1] + cam[3]
    return [int(u+0.5),int(v+0.5)]

def getRay(cam, uv):
    ray_uv = np.ones(3, dtype=np.float32)
    ray_uv[0] = (uv[0]-cam[2]) / cam[0]
    ray_uv[1] = (uv[1]-cam[3]) / cam[1]
    return np.expand_dims(ray_uv / LA.norm(ray_uv), -1)

def updateSeed(cam, seed, uv_cur, T_ref_cur):
    uv_ref = seed[0]
    z = depthFromTriangulation(cam, T_ref_cur, uv_ref, uv_cur)
    if 150<uv_ref[0]<154 and 160<uv_ref[1]<165:
        print(cam, T_ref_cur, uv_ref, uv_cur, z)
    if z>=0:
        tau = computeTau(cam, T_ref_cur, uv_ref, z)
        tau_inverse = 0.5 * (1.0/np.max([0.0000001, z-tau]) - 1.0/(z+tau))
        seed=updateSeedDist(seed, 1.0/z, tau_inverse*tau_inverse)
    return seed


################################ helpers for updateSeed ###########################################
def depthFromTriangulation(cam, T_ref_cur, uv_ref, uv_cur):
    T_cur_ref = LA.inv(T_ref_cur)
    R_cur_ref = T_cur_ref[0:3, 0:3]
    t_cur_ref = T_cur_ref[0:3, 3]
    ray_uv_ref = getRay(cam, uv_ref)
    ray_uv_cur = getRay(cam, uv_cur)

    A = np.hstack((np.matmul(R_cur_ref,ray_uv_ref), ray_uv_cur))
    AtA = np.matmul(A.T, A)
    if LA.det(AtA) < 1e-5:
        return -1
    depth2 = - np.matmul(np.matmul(LA.inv(AtA),A.T),t_cur_ref)
    depth = np.fabs(depth2[0])

    return depth


def computeTau(cam, T_ref_cur, uv_ref, z):
    px_error_angle = cam[4]
    ray_uv_ref = getRay(cam, uv_ref)
    t = np.expand_dims(T_ref_cur[0:3,3],-1)
    a = ray_uv_ref*z-t
    t_norm = LA.norm(t)
    a_norm = LA.norm(a)
    alpha = np.arccos(np.dot(ray_uv_ref.T,t)/t_norm) # dot product
    beta = np.arccos(np.dot(a.T,-t)/(t_norm*a_norm)) # dot product
    beta_plus = beta + px_error_angle
    gamma_plus = np.pi-alpha-beta_plus # triangle angles sum to PI
    z_plus = t_norm*np.sin(beta_plus)/np.sin(gamma_plus) # law of sines
    return (z_plus - z) # tau
    

def updateSeedDist(seed, x, tau2):
    [uv,a,b,mu,z_range,sigma2] = seed

    if mu<0:
        mu = x
        sigma2 = tau2
        seed = (uv,a,b,mu,z_range,sigma2)
    else:
        norm_scale = np.sqrt(sigma2 + tau2)
        if not np.isnan(norm_scale):
            s2 = 1.0/(1.0/sigma2 + 1.0/tau2)
            m = s2*(mu/sigma2 + x/tau2)
            C1 = a/(a+b) * norm.pdf(x, mu, norm_scale)
            C2 = b/(a+b) * 1.0/z_range
            normalization_constant = C1 + C2
            C1 /= normalization_constant
            C2 /= normalization_constant
            f = C1*(a+1.0)/(a+b+1.0) + C2*a/(a+b+1.0)
            e = C1*(a+1.0)*(a+2.0)/((a+b+1.0)*(a+b+2.0)) \
            + C2*a*(a+1.0)/((a+b+1.0)*(a+b+2.0))

            # update parameters
            mu_new = C1*m+C2*mu
            sigma2 = C1*(s2 + m*m) + C2*(sigma2 + mu*mu) - mu_new*mu_new
            # print([x, mu, mu_new])
            mu = mu_new
            a = (e - f) / (f - e / f)
            b = a * (1.0 - f) / f
            seed = (uv,a,b,mu,z_range,sigma2)
    
    return seed
