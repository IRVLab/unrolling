from __future__ import division, absolute_import
import os
import numpy as np
import cv2
import csv

def draw_flow(img, flow):
    h, w = img.shape[:2]
    flow_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for v in range(h):
        for u in range(w):
            fu,fv = flow[v,u].T
            if(fu*fu+fv*fv>0.2):
                cv2.line(flow_vis, (u, v), (np.int32(u+fu+0.5), np.int32(v+fv+0.5)), (0, 255, 0))

    return flow_vis


class dataLoaderTUM():
    def __init__(self, data_path="/mnt/data2/unRolling_Shutter/Rolling_Shutter_TUM/", seq_no=1, out_res=(320, 256), load_flow=False):
        self.gs_folder = "cam0/images/"
        self.rs_folder = "cam1/images/"
        self.df_folder = "flow/" if load_flow else "disparity/"
        self.load_flow = load_flow
        self.out_res = out_res
        self.get_train_paths(data_path, seq_no)

    def get_train_paths(self, data_path, seq_no):
        # if None, train all together
        if seq_no not in range(10): 
            all_sets = ["seq"+str(i) for i in range(10)]
        else:
            all_sets = ["seq"+str(seq_no)]
        # get all paths
        self.all_gs_paths, self.all_rs_paths, self.all_df_paths, self.all_files_id = [], [], [], []
        for p in all_sets:
            train_dir = os.path.join(data_path, p)
            gs_paths, rs_paths, df_paths, files_id = self.get_paired_paths(train_dir)
            self.all_gs_paths += gs_paths
            self.all_rs_paths += rs_paths
            self.all_df_paths += df_paths
            self.all_files_id += files_id

        self.num_train = len(self.all_df_paths)
        print ("Loaded {0} pairs of image-paths for training".format(self.num_train)) 

    def get_paired_paths(self, data_dir):
        gs_files = sorted(os.listdir(os.path.join(data_dir, self.gs_folder))) 
        rs_files = sorted(os.listdir(os.path.join(data_dir, self.rs_folder)))
        df_files = os.listdir(os.path.join(data_dir, self.df_folder))
        files_id = sorted([int(file[:-4]) for file in df_files])

        df_ext = '.npy' if self.load_flow else '.csv'
        gs_paths, rs_paths, df_paths = [], [], []
        for fi in files_id:
            df_paths.append(os.path.join(data_dir, self.df_folder, str(fi)+df_ext))
            gs_paths.append(os.path.join(data_dir, self.gs_folder, gs_files[fi]))
            rs_paths.append(os.path.join(data_dir, self.rs_folder, rs_files[fi]))

        return (gs_paths, rs_paths, df_paths, files_id)

    def read_resize_img(self, path):
        img = cv2.imread(path, 0).astype(np.float)
        self.in_res = (img.shape[1], img.shape[0])
        img = cv2.resize(img, self.out_res)
        return img

    def read_resize_disp(self, path):
        disp = np.array(list(csv.reader(open(path), delimiter=","))).astype(np.float32)
        res_ratio = self.out_res[0] / self.in_res[0]
        disp *= res_ratio
        return disp

    def load_batch(self, i, batch_size=1):
        batch_gs = self.all_gs_paths[i*batch_size:(i+1)*batch_size]
        batch_rs = self.all_rs_paths[i*batch_size:(i+1)*batch_size]
        batch_df = self.all_df_paths[i*batch_size:(i+1)*batch_size]
        batch_fi = self.all_files_id[i*batch_size:(i+1)*batch_size]

        imgs_gs, imgs_rs, disp_flows = [], [], []
        for idx in range(len(batch_gs)): 
            img_gs = self.read_resize_img(batch_gs[idx])
            img_rs = self.read_resize_img(batch_rs[idx])
            imgs_gs.append(img_gs)
            imgs_rs.append(img_rs)
            if self.load_flow:
                flow = np.load(batch_df[idx])
                assert(flow.shape[1:2] == img_gs.shape[1:2])
                disp_flows.append(flow)
            else:
                disp = self.read_resize_disp(batch_df[idx])
                disp_flows.append(disp)

        imgs_gs = np.array(imgs_gs, dtype=np.uint8)
        imgs_rs = np.array(imgs_rs, dtype=np.uint8)
        disp_flows = np.array(disp_flows, dtype=np.float32)

        imgs_gs = np.expand_dims(imgs_gs, -1) # (b, h, w, 1)
        imgs_rs = np.expand_dims(imgs_rs, -1) # (b, h, w, 1)
        if not self.load_flow:
            disp_flows = np.expand_dims(disp_flows, -1) # (b, h, w, 1)
        return imgs_gs, imgs_rs, disp_flows, batch_fi



if __name__=="__main__":
    data_loader = dataLoaderTUM()
    imgs_gs, imgs_rs = data_loader.load_batch()
    cv2.imwrite('a.png', imgs_gs[0,:])






