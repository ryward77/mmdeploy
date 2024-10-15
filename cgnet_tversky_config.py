norm_cfg = dict(type='SyncBN', eps=0.001, requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[72.39239876, 82.90891754, 73.15835921],
    std=[1, 1, 1],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=254,
    size=(240, 240))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[72.39239876, 82.90891754, 73.15835921],
        std=[1, 1, 1],
        bgr_to_rgb=False,
        pad_val=0,
        seg_pad_val=254,
        size=(240, 240)),
    backbone=dict(
        type='CGNet',
        norm_cfg=dict(type='SyncBN', eps=0.001, requires_grad=True),
        in_channels=3,
        num_channels=(64, 128, 256),
        num_blocks=(3, 42),
        dilations=(1, 2),
        reductions=(4, 8),
        act_cfg=dict(type='RReLU')),
    decode_head=dict(
        type='FCNHead',
        in_channels=512,
        in_index=2,
        channels=512,
        num_convs=0,
        concat_input=False,
        dropout_ratio=0,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', eps=0.001, requires_grad=True),
        loss_decode=dict(
            type='TverskyLoss',
            class_weight=[1, 0],
            smooth=0,
            alpha=0.5,
            beta=0.5,
            loss_weight=1.5,
            ignore_index=253)),
    train_cfg=dict(sampler=None),
    test_cfg=dict(mode='whole'))
dataset_type = 'GlottisDataset'
data_root = 'data/real/'
crop_size = (240, 240)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(240, 240),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(240, 240), cat_max_ratio=0.75),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(240, 240)),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale_factor': 0.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 0.75,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.0,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.25,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.75,
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 0.0,
                        'direction': 'horizontal'
                    }, {
                        'type': 'RandomFlip',
                        'prob': 1.0,
                        'direction': 'horizontal'
                    }], [{
                        'type': 'LoadAnnotations'
                    }], [{
                        'type': 'PackSegInputs'
                    }]])
]
maskdir = 'masks_adjusted_backup_8bit_2color'
imgdir = 'images_filtered_20%_no_cords_backup'
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='GlottisDataset',
        data_root='data/real/',
        data_prefix=dict(
            img_path='images_filtered_20%_no_cords_backup/training',
            seg_map_path='masks_adjusted_backup_8bit_2color/gtFine/Train'),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomResize',
                scale=(240, 240),
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=(240, 240), cat_max_ratio=0.75),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='GlottisDataset',
        data_root='data/real/',
        data_prefix=dict(
            img_path='images_filtered_20%_no_cords_backup/validation',
            seg_map_path='masks_adjusted_backup_8bit_2color/gtFine/Train'),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(240, 240)),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='GlottisDataset',
        data_root='data/real/',
        data_prefix=dict(
            img_path='images_filtered_20%_no_cords_backup/testing',
            seg_map_path='masks_adjusted_backup_8bit_2color/gtFine/Train'),
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(240, 240)),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mFscore', 'mIoU'])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='Adagrad', lr=0.03)
optim_wrapper = dict(
    type='OptimWrapper', optimizer=dict(type='Adagrad', lr=0.03))
TOTAL_ITERATIONS = 60000
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        by_epoch=False,
        begin=0,
        end=60000)
]
total_iters = 60000
train_cfg = dict(type='IterBasedTrainLoop', max_iters=60000, val_interval=500)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        save_best='mIoU',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(
        type='SegVisualizationHook',
        draw=True,
        show=False,
        wait_time=0,
        interval=25))
launcher = 'none'
work_dir = './work_dirs/cgnet_fcn_4xb8-60k_glottis-512x1024'
