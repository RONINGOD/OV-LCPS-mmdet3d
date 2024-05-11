_base_ = [
    '../_base_/datasets/nuscenes_ov_panoptic_lpmix.py', '../_base_/models/pfc.py',
    '../_base_/default_runtime.py'
]

# optimizer
# This schedule is mainly used by models on nuScenes dataset

max_epochs = 40

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
num_classes = 13
model = dict(
    decode_head=dict(
        use_lable_weight = False,
        cal_sem_loss = True,
        use_pa_seg_loss = True,
        use_dice_loss=True,
        use_sem_loss=True,
        panoptic_use_sigmoid = False,
        num_decoder_layers=6,
        num_classes = num_classes,
        vision_clip_dim = 768,
        num_queries=128,
        embed_dims=256,
        cls_channels=(256, 256, 768),
        mask_channels=(256, 256, 256, 256, 256),
        thing_class=[1,2,3,4,5,6,7,8,9,10],
        stuff_class=[11,12,13,14,15,16],
        ignore_index=0,
        loss_cls=dict(
            type= 'mmdet.CrossEntropyLoss',
            use_sigmoid= True,
            ignore_index=0,
            reduction='mean',
            loss_weight= 1.0),
        # loss_cls=dict(
        #     type='mmdet.FocalLoss',
        #     use_sigmoid=True,
        #     gamma=4.0,
        #     alpha=0.25,
        #     loss_weight=1.0),
    ))


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
val_dataloader = dict(batch_size=2,num_workers=8, )
test_dataloader = dict(batch_size=1,num_workers=8, )
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
# python -m torch.distributed.launch --node_rank=0 --nproc_per_node=2 --master_port=29503 ./train.py configs/pfc/pfc_nuscenes.py --launcher pytorch