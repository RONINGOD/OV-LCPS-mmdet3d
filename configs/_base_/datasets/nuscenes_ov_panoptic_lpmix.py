# For SemanticKitti we usually do 19-class segmentation.
# For labels_map we follow the uniform format of MMDetection & MMSegmentation
# i.e. we consider the unlabeled class as the last one, which is different
# from the original implementation of some methods e.g. Cylinder3D.
dataset_type = '_NuScenesDataset'
data_root = '/home/coisini/data/nuscenes'
ann_root = '/home/coisini/data/nuscenes/nus_pkl'
clip_feature_root = '/home/coisini/data/nuscenes_openseg_features'
version_root = 'v1.0-mini' # v1.0-trainval
class_names = ['noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle', 'motorcycle',
'pedestrian', 'traffic_cone', 'trailer', 'truck', 'driveable_surface', 'other_flat', 'sidewalk',
'terrain', 'manmade', 'vegetation']
label_mapping_path = "configs/_base_/datasets/nuscenes.yaml"

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

metainfo = dict(
    classes=class_names, seg_label_mapping=learning_mapping, max_label=31)

input_modality = dict(use_lidar=True, use_camera=False,use_clip_feature=True)

data_prefix = dict(
            pts='samples/LIDAR_TOP',
            img='samples',
            pts_instance_mask='',
            pts_semantic_mask='',
            pts_panoptic_mask='')
# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/semantickitti/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0rc4
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

pre_transform = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='_LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_panoptic_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=1000,
        dataset_type='nuscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping', )]

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='_LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_panoptic_3d=True,
        with_base_novel=True,
        seg_3d_dtype='np.int32',
        seg_offset=1000,
        dataset_type='nuscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping', ),
    dict(type='_BaseNovelClassMapping',),

    dict(type='_Pack3DDetInputs', keys=['points', 'pts_semantic_mask', 'pts_instance_mask','seenmask',
                'pts_clip_features','pts_clip_mask','pts_clip_features','pts_clip_mask','category_overlapping_mask',
                'base_novel_mapping','base_novel_mapping_inv','thing_class','stuff_class','text_features','base_thing_class','base_stuff_class'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='_LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_panoptic_3d=True,
        with_base_novel=True,
        seg_3d_dtype='np.int32',
        seg_offset=1000,
        dataset_type='nuscenes',
        backend_args=backend_args),
    dict(type='PointSegClassMapping', ),
    dict(type='_Pack3DDetInputs', keys=['points', 'pts_semantic_mask', 'pts_instance_mask','seenmask',
                'pts_clip_features','pts_clip_mask','pts_clip_features','pts_clip_mask','category_overlapping_mask',
                'base_novel_mapping','base_novel_mapping_inv','thing_class','stuff_class','text_features','base_thing_class','base_stuff_class'])
]


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix = data_prefix,
            ann_file=ann_root+'/nuscenes_infos_train.pkl',
            pipeline=train_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            ignore_index=0,
            clip_feature_root = clip_feature_root,
            label_mapping_path = label_mapping_path,
            backend_args=backend_args)),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix = data_prefix,
            ann_file=ann_root+'/nuscenes_infos_val.pkl',
            pipeline=test_pipeline,
            metainfo=metainfo,
            modality=input_modality,
            ignore_index=0,
            test_mode=True,
            clip_feature_root = clip_feature_root,
            label_mapping_path = label_mapping_path,
            backend_args=backend_args)),
)

val_dataloader = test_dataloader

val_evaluator = dict(type='_PanopticSegMetric',
                    class_names = class_names,
                    thing_class_inds=[1,2,3,4,5,6,7,8,9,10],
                    stuff_class_inds=[11,12,13,14,15,16],
                    thing_novel_class_inds = [3,6,7],
                    stuff_novel_class_inds = [16],
                    min_num_points=15,
                    id_offset = 2**16,
                    dataset_type='nuscenes',
                    )
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
