import os
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

    def __init__(self, path, n, train):
        phase = "test"
        if train:
            phase = "test"

        if not os.path.isfile(f"{path}/files_{phase}.txt"):
            scenes = os.listdir(path)
            self.files = []
            for scene in scenes:
                scene_path = f"{path}/{scene}"
                if not os.path.isdir(scene_path):
                    continue

                for i in range(0, 1):#lets only 640 by 480 images trough
                    for j in range(0, 4):
                        self.files.append((f"{scene_path}/im{i}_{j}.npy", f"{scene_path}/disp{i}_{j}.npy"))
            random.shuffle(self.files)
            #todo: split these phases by scene!
            file_list_train = self.files[0:int((len(self.files) * 9) / 10)]
            file_list_test = self.files[int((len(self.files) * 9) / 10):]
            self.store_list(f"{path}/file_train.txt", file_list_train)
            self.store_list(f"{path}/file_test.txt", file_list_test)

        self.files = self.load_list(f"{path}/file_{phase}.txt")
        if len(self.files) < n:
            print("too few images!")
        self.files = self.files[0:n]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        im = np.load(self.files[idx][0])
        disp = np.load(self.files[idx][1])
        return {"im0": im, "disp0": disp}





