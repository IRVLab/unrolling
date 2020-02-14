import numpy as np
import cv2
import rosbag
import rospy
from cv_bridge import CvBridge, CvBridgeError

filename = "/home/jiawei/Desktop/dataset-seq1.bag"
cam0_topic = "/cam0/image_raw"
cam1_topic = "/cam1/image_raw"
save_dir = "/home/jiawei/Workspace/data/datasets/TUM/seq1"

# TUM parameters
imageSize = (1280, 1024)
K1 = np.matrix([[743.4286936207343, 0, 618.7186883884866], 
               [0, 743.5545205462922, 506.7275058699658],
               [0,0,1]])
D1 = np.array([0.023584346301328347, -0.006764098468377487, 0.010259071387776937, -0.0037561745737771414])
K2 = np.matrix([[739.1654756101043, 0, 625.826167006398], 
               [0, 739.1438452683457, 517.3370973594253],
               [0,0,1]])
D2 = np.array([0.019327620961435945, 0.006784242994724914, -0.008658628531456217, 0.0051893686731546585])
T12 = np.matrix([[0.9999898230675194, -0.0032149094951349666, 0.003165141123167508, -0.10923281531657063],
              [0.0032300187644357624, 0.9999833583589267, -0.0047801656858271774, -0.0005117044964461212],
              [-0.003149720649973306, 0.004790340503623662, 0.9999835658138021, 0.00010164421140626545],
              [0, 0, 0, 1]])
R = T12[np.ix_([0,1,2], [0,1,2])]
tvec = T12[np.ix_([0,1,2], [3])]

# Fisheye sterep rectify
R1,R2,P1,P2,Q = cv2.fisheye.stereoRectify(K1,D1,K2,D2,imageSize,R,tvec,0)
map11, map12 = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1, imageSize, cv2.CV_32F)
map21, map22 = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2, imageSize, cv2.CV_32F)

# Open bag file.
bridge = CvBridge()
img0_count = 0
img1_count = 0
with rosbag.Bag(filename, 'r') as bag:
    for topic, msg, t in bag.read_messages():
        if topic == cam0_topic or topic == cam1_topic:
            # Cvt to cv image
            try:
                cv_image = bridge.imgmsg_to_cv2(msg)
            except CvBridgeError, e:
                print e
            
            # Rectify image and get image name
            if topic == cam0_topic:
                cv_image_rect = cv2.remap(cv_image, map11, map12, cv2.INTER_LINEAR)
                image_name = save_dir+"/img0/"+"{0:0>4}".format(img0_count)+".png"
                img0_count = img0_count+1
            else:
                cv_image_rect = cv2.remap(cv_image, map21, map22, cv2.INTER_LINEAR)
                image_name = save_dir+"/img1/"+"{0:0>4}".format(img1_count)+".png"
                img1_count = img1_count+1

            # Save image
            print image_name
            cv2.imwrite(image_name, cv_image_rect)