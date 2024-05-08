dataset_type = '_NuScenesDataset'
data_root = '/home/coisini/data/nuscenes'
ann_root = '/home/coisini/data/nuscenes/nus_pkl'
version_root = 'v1.0-mini'
class_names = [
    'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]
learning_mapping = dict({
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
})
metainfo = dict(
    classes=[
        'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation'
    ],
    seg_label_mapping=dict({
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
    }),
    max_label=31)
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(
    pts='samples/LIDAR_TOP',
    img='samples',
    pts_instance_mask='',
    pts_semantic_mask='',
    pts_panoptic_mask='')
backend_args = None
pre_transform = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=None),
    dict(
        type='_LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_panoptic_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=1000,
        dataset_type='nuscenes',
        backend_args=None),
    dict(type='PointSegClassMapping')
]
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=None),
    dict(
        type='_LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_panoptic_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=1000,
        dataset_type='nuscenes',
        backend_args=None),
    dict(type='PointSegClassMapping'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=None),
    dict(
        type='_LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_panoptic_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=1000,
        dataset_type='nuscenes',
        backend_args=None),
    dict(type='PointSegClassMapping'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='_NuScenesDataset',
            data_root='/home/coisini/data/nuscenes',
            data_prefix=dict(
                pts='samples/LIDAR_TOP',
                img='samples',
                pts_instance_mask='',
                pts_semantic_mask='',
                pts_panoptic_mask=''),
            ann_file=
            '/home/coisini/data/nuscenes/nus_pkl/nuscenes_infos_train.pkl',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=4,
                    backend_args=None),
                dict(
                    type='_LoadAnnotations3D',
                    with_bbox_3d=False,
                    with_label_3d=False,
                    with_panoptic_3d=True,
                    seg_3d_dtype='np.int32',
                    seg_offset=1000,
                    dataset_type='nuscenes',
                    backend_args=None),
                dict(type='PointSegClassMapping'),
                dict(
                    type='Pack3DDetInputs',
                    keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
            ],
            metainfo=dict(
                classes=[
                    'noise', 'barrier', 'bicycle', 'bus', 'car',
                    'construction_vehicle', 'motorcycle', 'pedestrian',
                    'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                    'other_flat', 'sidewalk', 'terrain', 'manmade',
                    'vegetation'
                ],
                seg_label_mapping=dict({
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
                }),
                max_label=31),
            modality=dict(use_lidar=True, use_camera=False),
            ignore_index=0,
            backend_args=None)))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='_NuScenesDataset',
            data_root='/home/coisini/data/nuscenes',
            data_prefix=dict(
                pts='samples/LIDAR_TOP',
                img='samples',
                pts_instance_mask='',
                pts_semantic_mask='',
                pts_panoptic_mask=''),
            ann_file=
            '/home/coisini/data/nuscenes/nus_pkl/nuscenes_infos_val.pkl',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=4,
                    backend_args=None),
                dict(
                    type='_LoadAnnotations3D',
                    with_bbox_3d=False,
                    with_label_3d=False,
                    with_panoptic_3d=True,
                    seg_3d_dtype='np.int32',
                    seg_offset=1000,
                    dataset_type='nuscenes',
                    backend_args=None),
                dict(type='PointSegClassMapping'),
                dict(
                    type='Pack3DDetInputs',
                    keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
            ],
            metainfo=dict(
                classes=[
                    'noise', 'barrier', 'bicycle', 'bus', 'car',
                    'construction_vehicle', 'motorcycle', 'pedestrian',
                    'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                    'other_flat', 'sidewalk', 'terrain', 'manmade',
                    'vegetation'
                ],
                seg_label_mapping=dict({
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
                }),
                max_label=31),
            modality=dict(use_lidar=True, use_camera=False),
            ignore_index=0,
            test_mode=True,
            backend_args=None)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='_NuScenesDataset',
            data_root='/home/coisini/data/nuscenes',
            data_prefix=dict(
                pts='samples/LIDAR_TOP',
                img='samples',
                pts_instance_mask='',
                pts_semantic_mask='',
                pts_panoptic_mask=''),
            ann_file=
            '/home/coisini/data/nuscenes/nus_pkl/nuscenes_infos_val.pkl',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=4,
                    backend_args=None),
                dict(
                    type='_LoadAnnotations3D',
                    with_bbox_3d=False,
                    with_label_3d=False,
                    with_panoptic_3d=True,
                    seg_3d_dtype='np.int32',
                    seg_offset=1000,
                    dataset_type='nuscenes',
                    backend_args=None),
                dict(type='PointSegClassMapping'),
                dict(
                    type='Pack3DDetInputs',
                    keys=['points', 'pts_semantic_mask', 'pts_instance_mask'])
            ],
            metainfo=dict(
                classes=[
                    'noise', 'barrier', 'bicycle', 'bus', 'car',
                    'construction_vehicle', 'motorcycle', 'pedestrian',
                    'traffic_cone', 'trailer', 'truck', 'driveable_surface',
                    'other_flat', 'sidewalk', 'terrain', 'manmade',
                    'vegetation'
                ],
                seg_label_mapping=dict({
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
                }),
                max_label=31),
            modality=dict(use_lidar=True, use_camera=False),
            ignore_index=0,
            test_mode=True,
            backend_args=None)))
val_evaluator = dict(
    type='_PanopticSegMetric',
    thing_class_inds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    stuff_class_inds=[11, 12, 13, 14, 15, 16],
    min_num_points=15,
    id_offset=65536,
    dataset_type='nuscenes')
test_evaluator = dict(
    type='_PanopticSegMetric',
    thing_class_inds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    stuff_class_inds=[11, 12, 13, 14, 15, 16],
    min_num_points=15,
    id_offset=65536,
    dataset_type='nuscenes')
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
grid_shape = [480, 360, 32]
num_classes = 17
model = dict(
    type='_P3Former',
    data_preprocessor=dict(
        type='_Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=[480, 360, 32],
            point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],
            max_num_points=-1,
            max_voxels=-1)),
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(
        type='_Asymm3DSpconv',
        grid_size=[480, 360, 32],
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-05, momentum=0.1),
        more_conv=True,
        out_channels=256),
    decode_head=dict(
        type='_P3FormerHead',
        num_classes=17,
        num_queries=128,
        embed_dims=256,
        point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],
        assigner_zero_layer_cfg=dict(
            type='mmdet.HungarianAssigner',
            match_costs=[
                dict(
                    type='mmdet.FocalLossCost',
                    weight=1.0,
                    binary_input=True,
                    gamma=2.0,
                    alpha=0.25),
                dict(type='mmdet.DiceCost', weight=2.0, pred_act=True)
            ]),
        assigner_cfg=dict(
            type='mmdet.HungarianAssigner',
            match_costs=[
                dict(
                    type='mmdet.FocalLossCost',
                    gamma=4.0,
                    alpha=0.25,
                    weight=1.0),
                dict(
                    type='mmdet.FocalLossCost',
                    weight=1.0,
                    binary_input=True,
                    gamma=2.0,
                    alpha=0.25),
                dict(type='mmdet.DiceCost', weight=2.0, pred_act=True)
            ]),
        sampler_cfg=dict(type='_MaskPseudoSampler'),
        loss_mask=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_dice=dict(type='mmdet.DiceLoss', loss_weight=2.0),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=4.0,
            alpha=0.25,
            loss_weight=1.0),
        num_decoder_layers=6,
        cls_channels=(256, 256, 17),
        mask_channels=(256, 256, 256, 256, 256),
        thing_class=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        stuff_class=[11, 12, 13, 14, 15, 16],
        ignore_index=0),
    train_cfg=None,
    test_cfg=dict(mode='whole'))
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
lr = 0.0008
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0008, weight_decay=0.01))
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[24, 32],
        gamma=0.2)
]
custom_imports = dict(
    imports=[
        'p3former.backbones.cylinder3d',
        'p3former.data_preprocessors.data_preprocessor',
        'p3former.decode_heads.p3former_head', 'p3former.segmentors.p3former',
        'p3former.task_modules.samplers.mask_pseduo_sampler',
        'evaluation.metrics.panoptic_seg_metric',
        'datasets.semantickitti_dataset', 'datasets.nuscenes_dataset',
        'datasets.transforms.loading', 'datasets.transforms.transforms_3d',
        'datasets.transforms.formating'
    ],
    allow_failed_imports=False)
launcher = 'pytorch'
work_dir = './work_dirs/p3former_nuscenes'
