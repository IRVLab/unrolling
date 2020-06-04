from __future__ import absolute_import,division,print_function
import numpy as np
import os
import matplotlib.pyplot as plt
from data_loader import dataLoader

data_loader = dataLoader() 
accs = data_loader.loadTestingAcceleration()

unrollnet_errs = np.load(os.path.join(os.getcwd(), "test_results/errs_unrollnet.npy"))
depthvel_errs = np.load(os.path.join(os.getcwd(), "test_results/errs_depthvelnet.npy"))
cv_gt_errs = np.load(os.path.join(os.getcwd(), "test_results/errs_cv_gt.npy"))

acc_t_err = np.array([accs[:,0],unrollnet_errs,cv_gt_errs,depthvel_errs]).T
acc_t_err = np.array(sorted(acc_t_err.tolist()))
acc_r_err = np.array([accs[:,1],unrollnet_errs,cv_gt_errs,depthvel_errs]).T
acc_r_err = np.array(sorted(acc_r_err.tolist()))

# average error
total_count = acc_r_err.shape[0]
step = 200
ave_idx = []
ave_t_un,ave_t_cg,ave_t_dn = [],[],[]
ave_r_un,ave_r_cg,ave_r_dn = [],[],[]
cur_i = 0
while cur_i+step<total_count:
    ave_idx.append(cur_i)
    ave_t_un.append(np.mean(acc_t_err[cur_i:cur_i+step,1]))
    ave_t_cg.append(np.mean(acc_t_err[cur_i:cur_i+step,2]))
    ave_t_dn.append(np.mean(acc_t_err[cur_i:cur_i+step,3]))
    ave_r_un.append(np.mean(acc_r_err[cur_i:cur_i+step,1]))
    ave_r_cg.append(np.mean(acc_r_err[cur_i:cur_i+step,2]))
    ave_r_dn.append(np.mean(acc_r_err[cur_i:cur_i+step,3]))
    cur_i = cur_i+step
ave_idx.append(cur_i)
ave_t_un.append(np.mean(acc_t_err[cur_i:,1]))
ave_t_cg.append(np.mean(acc_t_err[cur_i:,2]))
ave_t_dn.append(np.mean(acc_t_err[cur_i:,3]))
ave_r_un.append(np.mean(acc_r_err[cur_i:,1]))
ave_r_cg.append(np.mean(acc_r_err[cur_i:,2]))
ave_r_dn.append(np.mean(acc_r_err[cur_i:,3]))

plt.figure()
plt.subplot(1,2,1)
plt.plot(acc_t_err[:,0], 'k', label="Trans. Acc. (m/s2)")
plt.plot(ave_idx, ave_t_un, 'r', label="UnrollNet")
plt.plot(ave_idx, ave_t_dn, 'g', label="DepthVelNet")
plt.plot(ave_idx, ave_t_cg, 'b', label="CV_GT")
plt.grid()
plt.xlabel('Sample')
plt.ylabel('EPE')
plt.ylim([0,3])
plt.legend(loc="upper left")
plt.subplot(1,2,2)
plt.plot(acc_r_err[:,0], 'k', label="Rot. Acc. (rad/s2)")
plt.plot(ave_idx, ave_r_un, 'r', label="UnrollNet")
plt.plot(ave_idx, ave_r_dn, 'g', label="DepthVelNet")
plt.plot(ave_idx, ave_r_cg, 'b', label="CV_GT")
plt.grid()
plt.xlabel('Sample')
plt.ylabel('EPE')
plt.ylim([0,3])
plt.legend(loc="upper left")


p = 1. * np.arange(len(unrollnet_errs)) / (len(unrollnet_errs) - 1)
unrollnet_errs_sorted = np.sort(unrollnet_errs)
depthvel_errs_sorted = np.sort(depthvel_errs)
cv_gt_errs_sorted = np.sort(cv_gt_errs)
plt.figure()
plt.plot(unrollnet_errs_sorted, p, 'r', label="UnrollNet")
plt.plot(depthvel_errs_sorted, p, 'g', label="DepthVelNet")
plt.plot(cv_gt_errs_sorted, p, 'b', label="CV_GT")
plt.grid()
plt.xlabel('EPE')
plt.ylabel('CDF')
plt.gca().set_xlim(left=0)
plt.ylim([0,1])
plt.legend(loc="lower right")

plt.show()
