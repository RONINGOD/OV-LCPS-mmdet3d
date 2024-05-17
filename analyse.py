import pickle
import numpy as np
import torch
import mmengine
import os

# data = mmengine.load('/home/coisini/data/nuscenes/nus_pkl/nuscenes_infos_val.pkl')
# print(data.keys())
# print(data['data_list'][0].keys())
# print(data['data_list'][0]['lidar_points'])
# print(data['data_list'][0]['images'])
# print(data['data_list'][0]['pts_panoptic_mask_path'])

# learning_mapping={
#   1: 0,
#   5: 0,
#   7: 0,
#   8: 0,
#   10: 0,
#   11: 0,
#   13: 0,
#   19: 0,
#   20: 0,
#   0: 0,
#   29: 0,
#   31: 0,
#   9: 1,
#   14: 2,
#   15: 3,
#   16: 3,
#   17: 4,
#   18: 5,
#   21: 6,
#   2: 7,
#   3: 7,
#   4: 7,
#   6: 7,
#   12: 8,
#   22: 9,
#   23: 10,
#   24: 11,
#   25: 12,
#   26: 13,
#   27: 14,
#   28: 15,
#   30: 16
# }

# # 创建一个新的字典，用于存储反转的映射
# inverse_mapping = {}

# # 遍历原字典的键值对
# for key, value in learning_mapping.items():
#     # 将键添加到新字典中对应值的列表里
#     inverse_mapping.setdefault(value, []).append(key)

# print(inverse_mapping)
# print(os.path.basename('/home/coisini/data/nuscenes/nus_pkl/nuscenes_infos_val.pkl'))
# data = np.load('/home/coisini/data/nuscenes_openseg_features/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.npz',allow_pickle=True)
# print(data)
import torch

# def find_indices_torch(total_unq, unq):
#     # 将张量按最后一个维度展平
#     total_unq_flat = total_unq.view(-1, total_unq.shape[-1])
#     unq_flat = unq.view(-1, unq.shape[-1])

#     # 对每个元素进行编码
#     total_unq_encoded = total_unq_flat.mul(10 ** torch.arange(total_unq_flat.shape[-1]).to(total_unq_flat.device).view(1, -1)).sum(-1)
#     print(total_unq_encoded)
#     unq_encoded = unq_flat.mul(10 ** torch.arange(unq_flat.shape[-1]).to(unq_flat.device).view(1, -1)).sum(-1)

#     # 使用torch.isin检查total_unq_encoded中的每个元素是否在unq_encoded中
#     mask = torch.isin(total_unq_encoded.unsqueeze(-1), unq_encoded.unsqueeze(0))

#     # 恢复原始形状
#     mask = mask.view(total_unq.shape[:-1])

#     return mask

# # 示例用法
# total_unq = torch.tensor([[471, 351, 13], [4, 5, 6], [7, 8, 9]])
# unq = torch.tensor([[4, 5, 6], [7, 8, 9]])
# unq_test =  torch.tensor([[4, 5, 6], [471, 351, 13], [10, 5, 6], [7, 8, 9],[4, 5, 6]])
# print(torch.unique(unq_test,sorted=True,dim=0,return_inverse=True,))
# print(torch.unique(unq_test,sorted=False,dim=0,return_inverse=True,))
# mask = find_indices_torch(total_unq, unq)
# print(mask)  # 输出: tensor([False,  True,  True])
# # from mmdet3d.registry import MODELS
# # print(MODELS.build(dict()))
# from mmcv.cnn import build_activation_layer, build_norm_layer
# norm_cfg = dict(type='BN1d', eps=1e-5, momentum=0.01)
# print(build_norm_layer(norm_cfg,768)[1])

# data_root = '/home/coisini/data/nuscenes_multiview_openseg_val'
# file_lsit = os.listdir(data_root)
# print(len(file_lsit))
# data = torch.load(os.path.join(data_root,file_lsit[0]))
# print(data['feat'].shape)
# print(data['mask_full'].shape)
# print(data['mask_full'].sum())

# # print(len(os.listdir('/home/coisini/data/nuscenes_2d/train')))
# file_path = '/home/coisini/data/nuscenes/nus_pkl/nuscenes_infos_val.pkl'
# with open(file_path, 'rb') as file:
#     # 使用pickle.load()方法读取数据
#     pkl_data = pickle.load(file)
# print(pkl_data)

# data = torch.load('/home/coisini/data/nuscenes_3d/val/0a0d6b8c2e884134a3b48df43d54c36a.pth')
# print(data[0].shape,data[2].shape)

# data = np.load('/home/coisini/data/nuscenes_openseg_features/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402928147847.npz')
# print(data['point_mask'].shape)

# from mmengine.config import Config
# cfg = Config.fromfile('configs/pfc/pfc_nuscenes.py')
# from mmdet3d.registry import MODELS
# model = MODELS.build(cfg.model)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print(count_parameters(model))
# data = np.load('/home/coisini/project/Ovlcps/0a1b4e0aa3824b0a96bafae7105c58cc.npz')
# print(data)

import torch
from mmcv.ops import DynamicScatter,dynamic_scatter
from mmcv.ops import Voxelization

voxel_size = (0.32, 0.32, 6)
voxel_size_t = torch.tensor(voxel_size, device='cuda:0')
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
pc_range = torch.tensor(point_cloud_range, device='cuda:0')
cluster_scatter = DynamicScatter(voxel_size, point_cloud_range, average_points=True)
features = torch.tensor([[68.9309, 31.5947,  0.8600], 
        [69.0586, 31.4590,  0.8594]], device='cuda:0')
coors = torch.tensor([[  0,   0, 332, 449],[  0,   0, 332, 449]], device='cuda:0', dtype=torch.int32)
voxel_mean, mean_coors = cluster_scatter(features, coors)
print(voxel_mean, mean_coors)
import torch_scatter
print(torch_scatter.scatter_mean(features,coors[:,1:].long()))
# make sure voxelization is right
# print((features - pc_range[None, :3]) // voxel_size_t[None,:])