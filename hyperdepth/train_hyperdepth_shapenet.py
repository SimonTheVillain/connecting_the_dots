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



params = hd.TrainParams(
  n_trees=4,
  max_tree_depth=16, #according to the paper?
  n_test_split_functions=50,
  n_test_thresholds=10,
  n_test_samples=4096,
  min_samples_to_split=16,
  min_samples_for_leaf=8)

n_disp_bins = 20
depth_switch = 0

row_from = 100
row_to = 132
n_train_samples = 1024 #in the paper "connecting the dots" 8192 images are used for training.
n_test_samples = 32

train_ims, train_disps = get_data(n_train_samples, row_from, row_to, True)
test_ims, test_disps = get_data(n_test_samples, row_from, row_to, False)

for tree_depth in [8]:#, 10, 12, 14, 16]: # probably best results are
  depth_switch = tree_depth - 4 # todo: according to the supplementary it should be tree_depth - 6!!!!!!

  prefix = f'td{tree_depth}_ds{depth_switch}'
  prefix = Path(f'./forests/{prefix}/')
  prefix.mkdir(parents=True, exist_ok=True)

  hd.train_forest(params, train_ims, train_disps, n_disp_bins=n_disp_bins, depth_switch=depth_switch, forest_prefix=str(prefix / 'fr'), row_from=15, row_to=16, n_threads=12)

  es = hd.eval_forest(test_ims, test_disps, n_disp_bins=n_disp_bins, depth_switch=depth_switch, forest_prefix=str(prefix / 'fr'), row_from=15, row_to=16)

  np.save(str(prefix / 'ta.npy'), test_disps)
  np.save(str(prefix / 'es.npy'), es)

  #todo: hide these plots?
  plt.figure();
  plt.subplot(2,1,1); plt.imshow(test_disps[0], vmin=0, vmax=4);
  plt.subplot(2,1,2); plt.imshow(es[0], vmin=0, vmax=4);
  plt.show()
