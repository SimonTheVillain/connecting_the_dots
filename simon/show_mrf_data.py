import scipy.io.matlab
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

data = scipy.io.loadmat("/home/simon/datasets/mrf/raw/base_long.mat")
print(data["base"])
data = np.array(data["base"])
print(data.shape)

data = cv2.resize(data, (int(data.shape[1]/10), int(data.shape[0]/10)))
cv2.imshow("base", data * 1.0/255.0)
#cv2.waitKey()

scenes = ["angel", "arch", "fox", "gargoyle", "lion"]
scene = "angel"
captures = 12

for i in range(1, captures + 1):
    data = scipy.io.loadmat(f"/home/simon/datasets/mrf/raw/{scene}/ir{str(i).zfill(2)}.mat")
    data = data["J"].astype(np.float32)# unfortunately this is not just twice the VGA resolution
    cv2.imshow("ir", data * 0.01)

    data = scipy.io.loadmat(f"/home/simon/datasets/mrf/raw/{scene}/rgb{str(i).zfill(2)}.mat")
    data = data["I"]
    data = data[:, :, [2, 1, 0]]# rgb to bgr?
    cv2.imshow("rgb", data)

    data = scipy.io.loadmat(f"/home/simon/datasets/mrf/raw/{scene}/depth{str(i).zfill(2)}.mat")
    data = data["D"] * 1.0/1000.0 #Is the depth in milimeters? or inverted?
    cv2.imshow("depth", data)
    #show_pcl(data)

    cv2.waitKey()

