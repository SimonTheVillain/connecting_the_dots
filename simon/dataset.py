import os
from pathlib import Path
import numpy as np
import torch.utils.data as data
import random




class ShuffledDataset(data.Dataset):

    def store_list(self, path, list):
        with open(path, 'w') as f:
            for item in list:
                f.write(f"{item[0]}, {item[1]}\n")

    def load_list(self, path):
        list = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1]
                line_split = line.split(", ")
                list.append((line_split[0], line_split[1]))
        return list

    def __init__(self, data_root, n, train):
        phase = "test"
        if train:
            phase = "test"
        self.data_type = "syn"

        sample_paths = sorted((Path(data_root) / self.data_type).glob('0*/'))

        train_paths = sample_paths[2 ** 10:]# the trailing 8192 scene
        test_paths = sample_paths[:2 ** 8] # the first 256 scenes

        if train:
            paths = train_paths
        else:
            paths = test_paths

        self.files = []
        for scene_path in paths:
            for i in range(0, 1):#lets only 640 by 480 images trough
                for j in range(0, 4):
                    self.files.append((f"{scene_path}/im{i}_{j}.npy", f"{scene_path}/disp{i}_{j}.npy"))
        random.shuffle(self.files)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = np.load(self.files[idx][0])
        disp = np.load(self.files[idx][1])
        #todo: augment the data!!!!!
        return {"im0": im, "disp0": disp}





