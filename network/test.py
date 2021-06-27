# Learning Rolling Shutter Correction from Real Data without Camera Motion Assumption
# Copyright (C) <2021> <Jiawei Mo, Md Jahidul Islam, Junaed Sattar>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# fmt: off
from __future__ import print_function, division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import time
import shutil
import numpy as np
import pandas as pd
import cv2

from DataLoader import TumDataSet
from RsDepthNet import RsDepthNet
from helpers import getFlow
from RsPoseNet import RsPoseNet
# fmt: on


def get_flows_pred(data, batch_size = 32):
    flows_pred = np.empty((0, rows, cols, 2))
    for i in range(0, data['size'], batch_size):
        rg = range(i, min(i+batch_size, data['size']))
        img = np.array([cv2.imread(data['image_paths'][j])
                        for j in rg]) / 255.0
        imu = np.array([data['imus'][j] for j in rg])
        pose_mat = model_rspose.model.predict([img, imu])['flow']
        depth_pred = model_depth.model.predict(img)
        flow_pred = getFlow(depth_pred, pose_mat, dataset.params).numpy()
        flows_pred = np.append(flows_pred, flow_pred, axis=0)

    return flows_pred


def rectify_imgs(data, flow, img_path):
    os.makedirs(img_path)
    for i in range(data['size']):
        flow_rs2gs = flow[i]
        size_1d = rows*cols
        ind_v, ind_u = np.indices((rows, cols), dtype='float32')

        # map: rs to gs
        uv_rs = np.stack((ind_u, ind_v), axis=-1)
        uv_gs = np.array(uv_rs + flow_rs2gs + 0.5, dtype='int')
        u_gs = np.clip(uv_gs[:, :, 0], 0, cols-1)
        v_gs = np.clip(uv_gs[:, :, 1], 0, rows-1)
        uv_gs_1d = v_gs * cols + u_gs
        uv_gs_1d = np.reshape(uv_gs_1d, (size_1d))

        # reverse the map: gs to rs
        flow_gs2rs_1d_u = np.full((size_1d), np.nan, dtype='float32')
        flow_gs2rs_1d_v = np.full((size_1d), np.nan, dtype='float32')
        flow_gs2rs_1d_u[uv_gs_1d] = np.reshape(-flow_rs2gs[:, :, 0], (size_1d))
        flow_gs2rs_1d_v[uv_gs_1d] = np.reshape(-flow_rs2gs[:, :, 1], (size_1d))
        flow_gs2rs = np.stack([np.reshape(flow_gs2rs_1d_u, (rows, cols)),
                               np.reshape(flow_gs2rs_1d_v, (rows, cols))], axis=-1)

        # 2d interpolation
        flow_gs2rs[:, :, 0] = pd.DataFrame(flow_gs2rs[:, :, 0]).interpolate()
        flow_gs2rs[:, :, 1] = pd.DataFrame(flow_gs2rs[:, :, 1]).interpolate()

        # reconstruct gs image
        map_u = ind_u + flow_gs2rs[:, :, 0]
        map_v = ind_v + flow_gs2rs[:, :, 1]
        img_rs = cv2.imread(data['image_paths'][i])
        img_gs = cv2.remap(img_rs, map_u, map_v, cv2.INTER_LINEAR)
        cv2.imwrite('{}{}.png'.format(img_path, i), img_gs)


if __name__ == "__main__":
    # load data
    dataset = TumDataSet(True, True)
    rows, cols = dataset.params['img_shape']

    # load depth model
    model_depth = RsDepthNet(dataset.params)
    model_depth.model.load_weights(os.path.join(
        os.getcwd(), 'checkpoints/model_depth.hdf5'))

    # load pose model
    model_rspose = RsPoseNet(dataset.params)
    model_rspose.model.load_weights(os.path.join(
        os.getcwd(), 'checkpoints/model_rspose_yy.hdf5'))

    save_path = 'results/'
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    seq_data = dataset.data['test']
    for seq, data in seq_data.items():
        print('\n\n\n******************** Sequence {} *********************'.format(seq))
        flows_gt = np.array([np.load(data['flow_paths'][j])
                             for j in range(data['size'])])
        errs_input = np.nanmean(
            np.sqrt(np.sum(np.square(flows_gt), axis=-1)), axis=(1, 2))
        print('Input EPE:            {:.3f}'.format(np.mean(errs_input)))

        # predict correction flow
        t0 = time.time()
        flows_pred = get_flows_pred(data)
        fps = data['size'] / (time.time() - t0)
        errs_pred = np.nanmean(
            np.sqrt(np.sum(np.square(flows_gt-flows_pred), axis=-1)), axis=(1, 2))
        print('Prediction:           {:.3f} x {:.1f}% @ {:.1f}FPS'.format(np.mean(
            errs_pred), 100*(np.count_nonzero(errs_input-errs_pred > 0))/data['size'], fps))

        # rectify images
        t0 = time.time()
        rectify_imgs(data, flows_pred, '{}{}/'.format(save_path, seq))
        fps = data['size'] / (time.time() - t0)
        print('Rectify Images:       {:.1f}FPS'.format(fps))
