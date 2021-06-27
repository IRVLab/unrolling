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

from __future__ import print_function, division
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2


def rectify_gs_imgs(save_path, resolution):
    gs_img_path = save_path+"cam1/images_gs/"
    if not os.path.exists(gs_img_path):
        os.makedirs(gs_img_path)

    cols, rows = resolution
    rs_img_path = save_path+"cam1/images/"
    flows_rs2gs_path = save_path+"cam1/flows_rs2gs/"
    count = len(os.listdir(flows_rs2gs_path))
    for i in tqdm(range(count)):
        fi = str(i)
        flow_rs2gs = np.load(flows_rs2gs_path + fi + '.npy')
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
        img_rs = cv2.imread(rs_img_path + fi + '.png')
        img_gs = cv2.remap(img_rs, map_u, map_v, cv2.INTER_LINEAR)
        cv2.imwrite('{}{}.png'.format(gs_img_path, i), img_gs)
