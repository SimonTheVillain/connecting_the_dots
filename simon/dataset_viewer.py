import os
import cv2
import numpy as np

path = "/home/simon/datasets/shapenet_rendered/syn"


scenes = os.listdir(path)
for scene in scenes:

    path_scene = f"{path}/{scene}"
    if not os.path.isdir(path_scene):
        continue
    for i in range(0, 4):
        for j in range(0, 4):
            file = f"{path_scene}/ambient{i}_{j}.npy"
            data = np.load(file)
            print(f"{file} resolution{data.shape}")
            cv2.imshow("ambient", data[0, :, :])

            file = f"{path_scene}/im{i}_{j}.npy"
            data = np.load(file)
            cv2.imshow("im", data[0, :, :])

            file = f"{path_scene}/disp{i}_{j}.npy"
            data = np.load(file)
            cv2.imshow("disp", data[0, :, :] * 0.1)

            file = f"{path_scene}/mask{i}_{j}.npy"
            data = np.load(file)
            cv2.imshow("mask", data[0, :, :])

            file = f"{path_scene}/grad{i}_{j}.npy"
            data = np.load(file)
            cv2.imshow("grad", data[0, :, :])
            cv2.waitKey()
