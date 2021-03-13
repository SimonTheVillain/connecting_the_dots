import numpy as np
import matplotlib.pyplot as plt
import cv2
#todo: open3d and usage of the depth intrinsics parameters used in
import open3d as o3d


def show_pcl(z):
    #K = [[567.6, 0, 324.7], [0, 570.2, 250.1]
    #bl = 0.075
    fx = 567.6
    fy = 570.2
    cxr = 324.7
    cyr = 250.1
    print(z.shape)
    pts = []
    for i in range(0, z.shape[0]):
        for j in range(0, z.shape[1]):
            y = i - cyr
            x = j - cxr
            depth = z[i, j]
            if 0 < depth < 5:
                pts.append([x * depth / fx, y * depth / fy, depth])
    xyz = np.array(pts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

#import h5py
#with h5py.File("/home/simon/datasets/mrf/additional_2019/2/table1.mat", 'r') as f:
#    keys = f.keys()
#table = scipy.io.loadmat("/home/simon/datasets/mrf/additional_2019/2/table1.mat")

data = np.load("/home/simon/datasets/shapenet_rendered/syn/00000000/im0_0.npy")
cv2.imshow("shit", data[0, :, :])
cv2.waitKey()
