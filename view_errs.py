from __future__ import absolute_import, division, print_function
import numpy as np
import os
import matplotlib.pyplot as plt
from data_loader import dataLoader

rot_weight = 10
num_anchor = 4
step = 200
colors = ['r', 'g', 'b', 'c']

# load data
data_loader = dataLoader()
accs = data_loader.loadTestingAcceleration()

errs = np.empty((num_anchor, accs.shape[0]))
for i in range(num_anchor):
    errs[i, :] = np.load(os.path.join(
        os.getcwd(), 'test_results/{}/{}/errs.npy'.format(rot_weight, i+1)))

# sort data by acceleration
acc_t_idx = np.argsort(accs[:, 0])
acc_r_idx = np.argsort(accs[:, 1])
acc_t = accs[acc_t_idx, 0]
acc_r = accs[acc_r_idx, 1]
total_count = acc_t_idx.shape[0]

# average error
plt.figure()
plt.subplot(1, 2, 1)
min_err, max_err = 100, -1
for anchor_i in range(num_anchor):
    acc_t_err = errs[anchor_i, acc_t_idx]
    ave_acc_t, ave_err_t = [], []
    for cur_i in range(total_count-step):
        ave_acc_t.append(np.mean(acc_t[cur_i:cur_i+step]))
        ave_err_t.append(np.mean(acc_t_err[cur_i:cur_i+step]))
    plt.plot(ave_acc_t, ave_err_t,
             colors[anchor_i], label='Anchor{}'.format(anchor_i+1))
    min_err = min([min_err, min(ave_err_t)])
    max_err = max([max_err, max(ave_err_t)])
plt.grid()
plt.xlabel('Trans. Acc. (m/s2)')
plt.ylabel('EPE')
plt.ylim([min_err, max_err])
plt.legend(loc='upper left')
plt.subplot(1, 2, 2)
for anchor_i in range(num_anchor):
    acc_r_err = errs[anchor_i, acc_r_idx]
    ave_acc_r, ave_err_r = [], []
    for cur_i in range(total_count-step):
        ave_acc_r.append(np.mean(acc_r[cur_i:cur_i+step]))
        ave_err_r.append(np.mean(acc_r_err[cur_i:cur_i+step]))
    plt.plot(ave_acc_r, ave_err_r,
             colors[anchor_i], label='Anchor{}'.format(anchor_i+1))
    min_err = min([min_err, min(ave_err_r)])
    max_err = max([max_err, max(ave_err_r)])
plt.grid()
plt.xlabel('Rot. Acc. (rad/s2)')
plt.ylabel('EPE')
plt.ylim([min_err, max_err])
plt.legend(loc='upper left')


p = 1. * np.arange(total_count) / (total_count - 1)
plt.figure()
for anchor_i in range(num_anchor):
    errs_sorted = np.sort(errs[anchor_i])
    plt.plot(errs_sorted, p, colors[anchor_i],
             label='Anchor{}'.format(anchor_i+1))
plt.grid()
plt.xlabel('EPE')
plt.ylabel('CDF')
plt.gca().set_xlim(left=0)
plt.ylim([0, 1])
plt.legend(loc='lower right')

plt.show()
