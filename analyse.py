import pickle
import numpy as np
import torch
import mmengine

data = mmengine.load('/home/coisini/data/nuscenes/nus_pkl/nuscenes_infos_val.pkl')
print(data.keys())
print(data['data_list'][0].keys())
print(data['data_list'][0]['lidar_points'])
print(data['data_list'][0]['images'])
print(data['data_list'][0]['pts_panoptic_mask_path'])

learning_mapping={
  1: 0,
  5: 0,
  7: 0,
  8: 0,
  10: 0,
  11: 0,
  13: 0,
  19: 0,
  20: 0,
  0: 0,
  29: 0,
  31: 0,
  9: 1,
  14: 2,
  15: 3,
  16: 3,
  17: 4,
  18: 5,
  21: 6,
  2: 7,
  3: 7,
  4: 7,
  6: 7,
  12: 8,
  22: 9,
  23: 10,
  24: 11,
  25: 12,
  26: 13,
  27: 14,
  28: 15,
  30: 16
}

# 创建一个新的字典，用于存储反转的映射
inverse_mapping = {}

# 遍历原字典的键值对
for key, value in learning_mapping.items():
    # 将键添加到新字典中对应值的列表里
    inverse_mapping.setdefault(value, []).append(key)

print(inverse_mapping)