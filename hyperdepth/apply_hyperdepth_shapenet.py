import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import sys
import os

import hyperdepth as hd

sys.path.append('../')
import simon.dataset

dataset_path = os.path.expanduser("~/datasets/shapenet_rendered")

def get_data(n, row_from, row_to, train):
  imsizes = [(480, 640)]
  focal_lengths = [160]
  dset = simon.dataset.ShuffledDataset(dataset_path, n, train)
  ims = np.empty((n, row_to-row_from, imsizes[0][1]), dtype=np.uint8)
  disps = np.empty((n, row_to-row_from, imsizes[0][1]), dtype=np.float32)

  # the disparity offset that is needed for absolute results
  disp_offset = np.arange(0, imsizes[0][1]) * 0.0
  disp_offset = np.expand_dims(disp_offset, axis=(0, 1))

  for idx in range(n):
    print(f'load sample {idx} train={train}')
    sample = dset[idx]
    ims[idx] = (sample['im0'][0, row_from:row_to] * 255).astype(np.uint8)
    disps[idx] = sample['disp0'][0, row_from:row_to] + disp_offset
  return ims, disps



height = 480

row_from = 0
row_to = height
n_test_samples = 32

test_ims, test_disps = get_data(n_test_samples, row_from, row_to, False)

es = hd.eval_forest(test_ims, test_disps, n_disp_bins=n_disp_bins, depth_switch=depth_switch,
                    forest_prefix=str(prefix / 'fr'), row_from=0, row_to=height)
plt.figure()
plt.subplot(2, 1, 1)
plt.imshow(test_disps[0], vmin=0, vmax=4)
plt.subplot(2, 1, 2)
plt.imshow(es[0], vmin=0, vmax=4)
plt.show()
sys.exit()
