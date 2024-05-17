from typing import List, Optional
import numpy as np
import torch
from torch import Tensor
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from mmdet3d.utils import OptConfigType
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
from mmcv.ops import DynamicScatter
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape
import torch.nn.functional as F

@MODELS.register_module(force=True)
class _VisionClipLiDAR(BaseModule):
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
                 lidar_backbone: ConfigType,
                 clip_backbone: ConfigType,
                 voxel_encoder: ConfigType,
                 norm_cfg = dict(type='BN1d', eps=1e-5, momentum=0.01),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.lidar_backbone = MODELS.build(lidar_backbone)
        self.clip_backbone = MODELS.build(clip_backbone)
        self.voxel_encoder = MODELS.build(voxel_encoder)

    def forward(self, batch_inputs:dict,batch_data_samples:dict) -> Tensor:
        """Forward pass."""
        clip_voxel_feats,voxel_coors = self.clip_backbone(batch_inputs,batch_data_samples)
        """Extract features from points."""
        lidar_encoded_feats = self.voxel_encoder(batch_inputs['voxels']['voxels'],
                                           batch_inputs['voxels']['coors'])
        lidar_voxel_feats, _= self.lidar_backbone(lidar_encoded_feats[0], lidar_encoded_feats[1],
                          len(batch_inputs['points']))
        voxel_feats = torch.cat([clip_voxel_feats,lidar_voxel_feats])
        batch_inputs['voxel_vision_features'] = clip_voxel_feats
        return voxel_feats,voxel_coors
