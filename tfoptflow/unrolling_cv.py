import os
import argparse
import cv2
import csv
import numpy as np
from numpy import linalg as LA

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for di in range(0,h,10):
        for dj in range(0,w,10):
            oj, oi = flow[di,dj].T
            if oi*oi+oj*oj>1.0: 
                cv2.line(vis, (dj, di), (np.int32(dj+oj+0.5), np.int32(di+oi+0.5)), (0, 255, 0))

    return vis

def backproj(x, y, Kinv):
    X = Kinv * np.matrix([x, y, 1]).T
    return X.item(0), X.item(1)

def calOFErr(flow, Kinv):
    h, w = flow.shape[:2]
    A = np.empty(shape=[0, 9])
    for di in range(0,h,10):
        for dj in range(0,w,10):
            oj, oi = flow[di,dj].T
            if oi*oi+oj*oj>1.0: 
                hxd, hyd = backproj(dj, di, Kinv)
                hxf, hyf = backproj(dj+oj, di+oi, Kinv)
                hxo, hyo = hxf-hxd, hyf-hyd
                a = [hyd-hyo, hxo-hxd, hyo*hxd-hxo*hyd, hxd*hxd, 2*hxd*hyd, 2*hxd, hyd*hyd, 2*hyd, 1]
                A = np.vstack((A, a))
    U, S, V = LA.svd(A)
    # print(V[:1])
    err = A*V[:1]
    return LA.norm(err) / len(err)


def main():
    parser = argparse.ArgumentParser(description="Extract Optical Flow and Calculate Error by Rolling Shutter.")
    parser.add_argument("--img_path", help="Path to image folder")
    parser.add_argument("--cam_param", help="Distorttion parameters.")
    args = parser.parse_args()

    with open(args.cam_param) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        intrin_param = next(csv_reader)

    intrin_param = np.array(intrin_param).astype(np.float)
    K = np.matrix([[intrin_param[0], 0, intrin_param[2]], [0, intrin_param[1], intrin_param[3]], [0, 0, 1]])
    Kinv = K.I

    if not os.path.exists(args.img_path):
        print('Path {} does not exists. Exiting.'.format(args.img_path))
        sys.exit(1)

    image_filenames = sorted(os.listdir(args.img_path))
    # for i in range(len(image_filenames)-1):
    for i in range(100):
        img_path1 = os.path.join(args.img_path, image_filenames[i])
        img_path2 = os.path.join(args.img_path, image_filenames[i+1])
        img1, img2 = cv2.imread(img_path1, 0), cv2.imread(img_path2, 0)
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 1, 25, 3, 7, 1.5, 0)
        cv2.imshow('flow', draw_flow(img2, flow))
        cv2.waitKey(1)

        err = calOFErr(flow, Kinv)
        print(err)

if __name__ == '__main__':
    main()