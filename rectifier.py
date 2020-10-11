from __future__ import print_function, division
import numpy as np
import cv2
from numpy import linalg as LA
from scipy.spatial.transform import Rotation, RotationSpline
from scipy import interpolate
import pandas as pd


class rectifier:
    def getGS2RSFlow(depth_rs, cam, anchors_t_r):
        num_anchor = len(anchors_t_r)
        h, w = depth_rs.shape[:2]
        flow_gs2rs = np.empty([h, w, 2], dtype=np.float32)
        flow_gs2rs[:] = np.nan

        tm = np.arange(num_anchor+1) / num_anchor
        ts, rs = [[0, 0, 0]], [[0, 0, 0]]
        for i in range(num_anchor):
            ts.append(list(anchors_t_r[i][:3]))
            rs.append(list(anchors_t_r[i][3:]))
        t_spline = interpolate.CubicSpline(tm, ts)
        R_spline = RotationSpline(tm, Rotation.from_rotvec(rs))

        K = np.array([[cam[0], 0, cam[2]], [0, cam[1], cam[3]], [0, 0, 1]])
        K_i = LA.inv(K)

        # Project from rs to gs
        for v_rs in range(h):
            tm = v_rs/(h-1)
            KRK_i = np.matmul(np.matmul(K, R_spline(tm).as_matrix()), K_i)
            Kt = np.matmul(K, t_spline(tm))
            for u_rs in range(w):
                if np.isnan(depth_rs[v_rs, u_rs]):
                    continue

                p_gs = depth_rs[v_rs, u_rs] * \
                    np.matmul(KRK_i, np.array([u_rs, v_rs, 1])) + Kt
                u_gs, v_gs = p_gs[0] / p_gs[2], p_gs[1] / p_gs[2]
                if not np.isnan(u_gs):
                    u_gsi, v_gsi = int(u_gs+0.5), int(v_gs+0.5)
                    if 0 <= u_gsi < w and 0 <= v_gsi < h:
                        flow_gs2rs[v_gsi, u_gsi, 0] = u_rs-u_gs
                        flow_gs2rs[v_gsi, u_gsi, 1] = v_rs-v_gs

        return flow_gs2rs

    def rectifyImgByFlow(img, flow):
        h, w = img.shape[:2]
        indy, indx = np.indices((h, w), dtype=np.float32)
        flow_u = pd.DataFrame(flow[:, :, 0])
        flow_v = pd.DataFrame(flow[:, :, 1])
        flow_interp = np.empty_like(flow)
        flow_interp[:, :, 0] = flow_u.interpolate(
            method='linear', limit_direction='forward', axis=0)
        flow_interp[:, :, 1] = flow_v.interpolate(
            method='linear', limit_direction='forward', axis=0)
        map_x = indx.reshape(h, w).astype(np.float32) + flow_interp[:, :, 0]
        map_y = indy.reshape(h, w).astype(np.float32) + flow_interp[:, :, 1]
        img_rectified = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        return img_rectified
