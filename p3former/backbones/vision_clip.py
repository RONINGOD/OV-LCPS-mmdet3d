from typing import List, Optional
import numpy as np
import torch
from torch import Tensor
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule,constant_init
from mmdet3d.utils import OptConfigType
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
from mmcv.ops import DynamicScatter
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
import torch.nn.functional as F
import torch_scatter

@MODELS.register_module(force=True)
class _VisionClip(BaseModule):
    """Asymmetrical 3D convolution networks.

    Args:
        grid_size (int): Size of voxel grids.
        input_channels (int): Input channels of the block.
        base_channels (int): Initial size of feature channels before
            feeding into Encoder-Decoder structure. Defaults to 16.
        backbone_depth (int): The depth of backbone. The backbone contains
            downblocks and upblocks with the number of backbone_depth.
        height_pooing (List[bool]): List indicating which downblocks perform
            height pooling.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layer. Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01)).
        init_cfg (dict, optional): Initialization config.
            Defaults to None.
    """

    def __init__(self,
                 voxel_layer:OptConfigType = None,
                 scatter_points_mode: bool = 'max',
                 scatter_type:str='dynamic',
                 clip_vision_dim :int =768,
                 norm_cfg = dict(type='BN1d', eps=1e-5, momentum=0.01),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.voxel_layer = VoxelizationByGridShape(**voxel_layer)
        self.scatter_type =scatter_type
        if self.scatter_type=='dynamic':
            self.vfe_scatter = DynamicScatter(self.voxel_layer.voxel_size,
                                    self.voxel_layer.point_cloud_range,
                                    ( scatter_points_mode!= 'max'))
                
        self.clip_vision_dim = clip_vision_dim
        self.pre_norm = build_norm_layer(norm_cfg,clip_vision_dim)[1]
        self.init_weights()
        
    def init_weights(self):
        if self.pre_norm is not None:
            constant_init(self.pre_norm, val=1, bias=0)


    def forward(self, batch_inputs:dict,batch_data_samples:dict) -> Tensor:
        """Forward pass."""
        clip_features = batch_inputs['pts_clip_features'] # [34752, 768]
        point_mask = batch_inputs['pts_clip_mask']
        coors = batch_inputs['voxels']['coors']
        points = batch_inputs['points']
        B = len(clip_features)
        voxels = []
        for batch in range(B):
            point_voxel_fea = clip_features[batch].new_ones([points[batch].shape[0],self.clip_vision_dim])*1e-5
            point_voxel_fea[point_mask[batch]] = clip_features[batch]
            voxels.append(point_voxel_fea)
            
        voxels = torch.cat(voxels,dim=0)
        if self.pre_norm is not None:
            voxels  = self.pre_norm(voxels)
        if self.scatter_type=='dynamic':
            voxel_feats, voxel_coors = self.vfe_scatter(voxels, coors)
        else:
            cur_device = coors[0].get_device()
            pt_num = voxels.shape[0]
            # shuffle the data
            shuffled_ind = torch.randperm(pt_num, device=cur_device)
            voxels = voxels[shuffled_ind, :]
            coors = coors[shuffled_ind, :]
            # unique xy grid index
            voxel_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
            voxel_coors = voxel_coors.long()
            voxel_feats = torch_scatter.scatter_mean(voxels, unq_inv, dim=0)

        return voxel_feats,voxel_coors
