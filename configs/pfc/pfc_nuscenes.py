_base_ = [
    '../_base_/datasets/nuscenes_ov_panoptic_lpmix.py',
    '../_base_/default_runtime.py'
]

# optimizer
# This schedule is mainly used by models on nuScenes dataset

max_epochs = 200
find_unused_parameters = False
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
num_classes = 13

grid_shape = [480, 360, 32]
num_classes = 13
point_cloud_range = [0, -3.14159265359, -5, 50, 3.14159265359, 3]  # nuscenes z[-5,3]
norm_cfg = dict(type='BN1d', eps=1e-5, momentum=0.01)
model = dict(
    type='_PFC',
    data_preprocessor=dict(
        type='_Det3DDataPreprocessor',
        voxel=True,
        open_vocabulary = True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=point_cloud_range,
            max_num_points=-1,
            max_voxels=-1,
        ),
    ),
    # voxel_encoder=dict(
    #     type='SegVFE',
    #     feat_channels=[64, 128, 256, 256],
    #     in_channels=6,
    #     with_voxel_center=True,
    #     feat_compression=16,
    #     return_point_feats=False),
    backbone=dict(
        type='_VisionClip',
        scatter_type='hard',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=point_cloud_range,
            max_num_points=-1,
            max_voxels=-1,
        ),
        scatter_points_mode='avg',
        clip_vision_dim = 768,
        ),
    decode_head=dict(
        type='_PFCHead', # _PFCHeadQuery _PFCHeadQueryPoints
        geometric_ensemble_alpha=0.0,
        geometric_ensemble_beta=1.0,
        use_lable_weight = False,
        cal_sem_loss = True,
        use_pa_seg_loss = True,
        use_dice_loss=True,
        use_sem_loss=True,
        panoptic_use_sigmoid = False,
        num_decoder_layers=6,
        num_classes = num_classes,
        vision_clip_dim = 768,
        num_queries=256,
        embed_dims=256,
        cls_channels=(256, 256, 768),
        mask_channels=(256, 256, 256, 256, 256),
        thing_class=[1,2,3,4,5,6,7,8,9,10],
        stuff_class=[11,12,13,14,15,16],
        ignore_index=0,
        point_cloud_range=point_cloud_range,
        assigner_zero_layer_cfg=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                        dict(type='mmdet.FocalLossCost', weight=1.0, binary_input=True, gamma=2.0, alpha=0.25),
                        dict(type='mmdet.DiceCost', weight=2.0, pred_act=True),
                    ]),
        assigner_cfg=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                        dict(type='mmdet.FocalLossCost', gamma=4.0,alpha=0.25,weight=1.0),
                        dict(type='mmdet.FocalLossCost', weight=1.0, binary_input=True, gamma=2.0, alpha=0.25),
                        dict(type='mmdet.DiceCost', weight=2.0, pred_act=True),
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
            type= 'mmdet.CrossEntropyLoss',
            use_sigmoid= True,
            ignore_index=0,
            loss_weight= 2.0),
        # loss_cls=dict(
        #     type='mmdet.FocalLoss',
        #     use_sigmoid=True,
        #     gamma=4.0,
        #     alpha=0.25,
        #     loss_weight=1.0),
    ),
    train_cfg=None,
    test_cfg=dict(mode='whole'),
)

lr = 0.0008
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))


train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[24, 32],
        gamma=0.2)
]

train_dataloader = dict(batch_size=2,num_workers=8, )
val_dataloader = dict(batch_size=4,num_workers=8, )
test_dataloader = dict(batch_size=4,num_workers=8, )
# test_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=5,
#         use_dim=4,
#         backend_args=None),
#     dict(type='_Pack3DDetInputs', keys=['points', 'lidar_path'])
# ]

# test_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='RepeatDataset',
#         times=1,
#         dataset=dict(
#             pipeline=test_pipeline,
#             ann_file='semantickitti_infos_mini.pkl'))
# )

# test_evaluator = dict(submission_prefix='semantickitti_submission')

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))

custom_imports = dict(
    imports=[
        'p3former.backbones.cylinder3d',
        'p3former.backbones.vision_clip',
        'p3former.data_preprocessors.data_preprocessor',
        'p3former.decode_heads.p3former_head',
        'p3former.decode_heads.pfc_head',
        'p3former.decode_heads.pfc_head_query',
        'p3former.decode_heads.pfc_head_query_points',
        'p3former.segmentors.pfc',
        'p3former.task_modules.samplers.mask_pseduo_sampler',
        'evaluation.metrics.panoptic_seg_metric',
        'datasets.semantickitti_dataset',
        'datasets.nuscenes_dataset',
        'datasets.transforms.loading',
        'datasets.transforms.transforms_3d',
        'datasets.transforms.formating',
    ],
    allow_failed_imports=False)
# python -m torch.distributed.launch --node_rank=0 --nproc_per_node=2 --master_port=29503 ./train.py configs/pfc/pfc_nuscenes_cross.py --launcher pytorch