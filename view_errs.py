from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from data_loader import dataLoader

# read num_anchor from command line
parser = argparse.ArgumentParser()
parser.add_argument(
    '--folder', help='Folder to show: test or seq2', default='test')
args = parser.parse_args()
folder = args.folder
print('Folder: {}'.format(folder))

# parameters
step = 100
num_anchor = [1, 2, 4, 8, 16]
colors = ['r', 'g', 'b', 'c', 'k']

# load data
data_loader = dataLoader()
if folder == 'test':
    accs = data_loader.loadTestingAcceleration()
else:
    accs = data_loader.loadSeqAcceleration()

# sort data by acceleration
acc_sorted_idx = np.argsort(accs)
acc_sorted = accs[acc_sorted_idx]
total_count = acc_sorted_idx.shape[0]

# load errors
errs = []
for anchor_i in num_anchor:
    err = np.load(os.path.join(
        os.getcwd(), 'test_results/{}/errs{}.npy'.format(folder, anchor_i)))
    print('Anchor: {}; EPE errs: {:.3f}'.format(anchor_i, np.mean(err)))
    errs.append(err)

# error against acceleration
plt.figure()
min_err, max_err = 100, -1
for anchor_i in range(len(num_anchor)):
    err_sorted = errs[anchor_i][acc_sorted_idx]
    ave_acc, ave_err = [], []
    for cur_i in range(total_count-step):
        ave_acc.append(np.mean(acc_sorted[cur_i:cur_i+step]))
        ave_err.append(np.mean(err_sorted[cur_i:cur_i+step]))
    plt.plot(ave_acc, ave_err, colors[anchor_i],
             label='Anchor{}'.format(num_anchor[anchor_i]))
    min_err = min([min_err, min(ave_err)])
    max_err = max([max_err, max(ave_err)])
plt.grid()
plt.xlabel(
    'Acc: {} Trans.(m/s2) + Rot.(rad/s2)'.format(data_loader.trans_weight))
plt.ylabel('EPE')
plt.ylim([min_err, max_err])
plt.legend(loc=('lower right' if folder == 'test' else 'upper left'))

# cdf
plt.figure()
p = 1. * np.arange(total_count) / (total_count - 1)
for anchor_i in range(len(num_anchor)):
    errs_sorted = np.sort(errs[anchor_i])
    plt.plot(errs_sorted, p, colors[anchor_i],
             label='Anchor{}'.format(num_anchor[anchor_i]))
plt.grid()
plt.xlabel('EPE')
plt.ylabel('CDF')
plt.gca().set_xlim(left=0)
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
