# Copyright (c) OpenMMLab. All rights reserved.
# Modified from mmdetection3d.
from typing import Callable, List, Optional, Union

import numpy as np

from mmdet3d.registry import DATASETS
from mmdet3d.datasets.seg3d_dataset import Seg3DDataset
import yaml
from os import path as osp

def transform_map(class_list):
    return {i:class_list[i] for i in range(len(class_list))}

def inverse_transform(learning_map):
    return {v: k for k, v in learning_map.items()} 

@DATASETS.register_module(force=True)
class _NuScenesDataset(Seg3DDataset):
    r"""SemanticKitti Dataset.

    This class serves as the API for experiments on the SemanticKITTI Dataset
    Please refer to <http://www.semantic-kitti.org/dataset.html>`_
    for data downloading

    Args:
        data_root (str, optional): Path of dataset root. Defaults to None.
        ann_file (str): Path of annotation file. Defaults to ''.
        metainfo (dict, optional): Meta information for dataset, such as class
            information. Defaults to None.
        data_prefix (dict): Prefix for training data. Defaults to
            dict(pts='',
                 img='',
                 pts_instance_mask='',
                 pts_semantic_mask='').
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input,
            it usually has following keys:

                - use_camera: bool
                - use_lidar: bool
            Defaults to dict(use_lidar=True, use_camera=False).
        ignore_index (int, optional): The label index to be ignored, e.g.
            unannotated points. If None is given, set to len(self.classes) to
            be consistent with PointSegClassMapping function in pipeline.
            Defaults to None.
        scene_idxs (np.ndarray or str, optional): Precomputed index to load
            data. For scenes with many points, we may sample it several times.
            Defaults to None.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
    """
    METAINFO = {
        'classes': ('noise','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
                    'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
                    'terrain', 'manmade', 'vegetation'),
        'palette': [[0,0,0], [220,220,  0], [119, 11, 32],
                    [0, 60, 100], [0, 0, 250], [230,230,250],
                    [0, 0, 230], [220, 20, 60], [250, 170, 30],
                    [200, 150, 0], [0, 0, 110], [128, 64, 128], [0,250, 250],
                    [244, 35, 232], [152, 251, 152], [70, 70, 70], [107,142, 35]],
        'seg_valid_class_ids':
        tuple(range(17)),
        'seg_all_class_ids':
        tuple(range(17)),
    }

    def __init__(self,
                 data_root: Optional[str] = None,
                 ann_file: str = '',
                 metainfo: Optional[dict] = None,
                 data_prefix: dict = dict(
                     pts='',
                     img='',
                     pts_instance_mask='',
                     pts_semantic_mask=''),
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False,use_clip_feature=False),
                 ignore_index: Optional[int] = None,
                 clip_feature_root: str = '',
                 label_mapping_path: str = '',
                 scene_idxs: Optional[Union[str, np.ndarray]] = None,
                 test_mode: bool = False,
                 **kwargs) -> None:
        if modality['use_clip_feature']:
            self.clip_feature_root = clip_feature_root
            self.unseen_class = []
            self.label_mapping_path = label_mapping_path
            self.test_mode = test_mode
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            metainfo=metainfo,
            data_prefix=data_prefix,
            pipeline=pipeline,
            modality=modality,
            ignore_index=ignore_index,
            scene_idxs=scene_idxs,
            test_mode=test_mode,
            **kwargs)


    def get_base_novel_mapping(self,label_mapping_path):
        with open(label_mapping_path, 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.thing_list = [cl for cl, is_thing in nuscenesyaml['thing_class'].items() if is_thing]
        self.stuff_list = [cl for cl, is_stuff in nuscenesyaml['stuff_class'].items() if is_stuff]
        self.base_thing_list = [cl for cl, is_thing in nuscenesyaml['base_thing_class'].items() if is_thing]
        self.base_stuff_list = [cl for cl, is_stuff in nuscenesyaml['base_stuff_class'].items() if is_stuff]
        self.novel_thing_list = [cl for cl, is_thing in nuscenesyaml['novel_thing_class'].items() if is_thing]
        self.novel_stuff_list = [cl for cl, is_stuff in nuscenesyaml['novel_stuff_class'].items() if is_stuff]
        base_novel_label_mapping = transform_map(np.hstack([0,self.base_thing_list,self.base_stuff_list,self.novel_thing_list,self.novel_stuff_list]))
        base_novel_label_mapping_inv = inverse_transform(base_novel_label_mapping)
        if not self.test_mode:
            self.thing_list = self.base_thing_list
            self.text_features = np.load(osp.join(self.clip_feature_root,'base_text_features.npy'))
            # self.text_features = np.load(osp.join(self.clip_feature_root,'base32_text_features.npy'))
            self.stuff_list = self.base_stuff_list
            self.unseen_class = list(set(range(1,len(nuscenesyaml['thing_class'].items()))).difference(set(self.base_stuff_list+self.base_thing_list)))
        else:
            self.text_features = np.load(osp.join(self.clip_feature_root,'total_text_features.npy'))
            # self.text_features = np.load(osp.join(self.clip_feature_root,'total44_text_features.npy'))
        self.thing_class = np.sort(np.vectorize(base_novel_label_mapping_inv.__getitem__)(self.thing_list))
        self.stuff_class = np.sort(np.vectorize(base_novel_label_mapping_inv.__getitem__)(self.stuff_list))
        self.total_class = np.sort(np.vectorize(base_novel_label_mapping_inv.__getitem__)(np.hstack([0,self.base_thing_list+self.base_stuff_list])))
        self.category_overlapping_mask = np.hstack((np.full(1,True,dtype=bool),np.full(len(self.base_thing_list+self.base_stuff_list), True, dtype=bool),np.full(len(self.novel_thing_list+self.novel_stuff_list),False,dtype=bool)))
        seg_base_novel_label_mapping = np.zeros(len(base_novel_label_mapping.keys()), dtype=np.int64)
        seg_base_novel_label_mapping_inv = np.zeros(len(base_novel_label_mapping_inv.keys()), dtype=np.int64)

        for k,v in base_novel_label_mapping.items():
            seg_base_novel_label_mapping[k] = v
        for k,v in base_novel_label_mapping_inv.items():
            seg_base_novel_label_mapping_inv[k] = v
        return seg_base_novel_label_mapping,seg_base_novel_label_mapping_inv

    def get_seg_label_mapping(self, metainfo):
        if self.modality['use_clip_feature']:
            self.base_novel_mapping,self.base_novel_mapping_inv = self.get_base_novel_mapping(self.label_mapping_path)
        seg_label_mapping = np.zeros(metainfo['max_label'] + 1, dtype=np.int64)
        for idx in metainfo['seg_label_mapping']:
            seg_label_mapping[idx] = metainfo['seg_label_mapping'][idx]
        return seg_label_mapping

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Convert all relative path of needed modality data file to
        the absolute path. And process
        the `instances` field to `ann_info` in training stage.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts', ''),
                    info['lidar_points']['lidar_path'])
            if 'num_pts_feats' in info['lidar_points']:
                info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']

        if self.modality['use_camera']:
            for cam_id, img_info in info['images'].items():
                if 'img_path' in img_info:
                    img_info['img_path'] = osp.join(
                        self.data_prefix.get('img', ''), cam_id, img_info['img_path'])

        if self.modality['use_clip_feature']:
            lidar_name = info['lidar_path'].split('.')[0]
            lidar_name = lidar_name.split(self.data_root)[1][1:]
            info['clip_feature_path'] = osp.join(self.clip_feature_root,lidar_name+'.npz')
            info['thing_class'] = self.thing_class
            info['stuff_class'] = self.stuff_class
            info['total_class'] = self.total_class
            info['category_overlapping_mask'] = self.category_overlapping_mask
            info['base_novel_mapping'] = self.base_novel_mapping
            info['base_novel_mapping_inv'] = self.base_novel_mapping_inv
            info['unseen_class'] = self.unseen_class
            info['text_features'] = self.text_features

        if 'pts_instance_mask_path' in info:
            info['pts_instance_mask_path'] = \
                osp.join(self.data_prefix.get('pts_instance_mask', ''),
                         info['pts_instance_mask_path'])

        if 'pts_semantic_mask_path' in info:
            info['pts_semantic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_semantic_mask', ''),
                         info['pts_semantic_mask_path'])

        # only be used in `PointSegClassMapping` in pipeline
        # to map original semantic class to valid category ids.
        info['seg_label_mapping'] = self.seg_label_mapping

        # 'eval_ann_info' will be updated in loading transforms
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = dict()
        if 'pts_panoptic_mask_path' in info:
            info['pts_panoptic_mask_path'] = \
                osp.join(self.data_prefix.get('pts_panoptic_mask', ''),
                         info['pts_panoptic_mask_path'])

        return info
