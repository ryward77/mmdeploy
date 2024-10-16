class_weight = [
    2,
    0,
]
crop_size = (
    240,
    240,
)
data_preprocessor = dict(
    bgr_to_rgb=False,
    mean=[
        72.39239876,
        82.90891754,
        73.15835921,
    ],
    pad_val=0,
    seg_pad_val=254,
    size=(
        240,
        240,
    ),
    std=[
        1,
        1,
        1,
    ],
    type='SegDataPreProcessor')
data_root = 'data/real/'
dataset_type = 'GlottisDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=4800,
        rule='greater',
        save_best='mFscore',
        type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True,
        interval=25,
        show=False,
        type='SegVisualizationHook',
        wait_time=0))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
]
imgdir = 'images_filtered_20%_no_cords_backup'
iters = 120000
launcher = 'none'
load_from = 'work_dirs/ddrnet_23-slim_in1k-pre_2xb6-120k_glottis/best_mFscore_iter_97200.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
maskdir = 'masks_adjusted_backup_8bit_2color'
model = dict(
    backbone=dict(
        align_corners=False,
        channels=32,
        in_channels=3,
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        ppm_channels=128,
        type='DDRNet'),
    data_preprocessor=dict(
        bgr_to_rgb=False,
        mean=[
            72.39239876,
            82.90891754,
            73.15835921,
        ],
        pad_val=0,
        seg_pad_val=254,
        size=(
            240,
            240,
        ),
        std=[
            1,
            1,
            1,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=64,
        dropout_ratio=0.0,
        in_channels=128,
        loss_decode=[
            dict(
                alpha=0.9,
                beta=0.1,
                class_weight=[
                    2,
                    0,
                ],
                ignore_index=253,
                loss_weight=2,
                smooth=0,
                type='TverskyLoss'),
            dict(
                alpha=0.9,
                beta=0.1,
                class_weight=[
                    2,
                    0,
                ],
                ignore_index=253,
                loss_weight=0.8,
                smooth=0,
                type='TverskyLoss'),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        type='DDRHead'),
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=None,
    optimizer=dict(lr=0.003, type='AdamW', weight_decay=0.001),
    type='OptimWrapper')
optimizer = dict(lr=0.003, type='AdamW', weight_decay=0.001)
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=120000,
        eta_min=0,
        power=0.9,
        type='PolyLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images_filtered_20%_no_cords_backup/testing',
            seg_map_path='masks_adjusted_backup_8bit_2color/gtFine/Train'),
        data_root='data/real/',
        img_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                240,
                240,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='GlottisDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=(
        240,
        240,
    ), type='Resize'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(
    max_iters=120000, type='IterBasedTrainLoop', val_interval=2400)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_prefix=dict(
            img_path='images_filtered_20%_no_cords_backup/training',
            seg_map_path='masks_adjusted_backup_8bit_2color/gtFine/Train'),
        data_root='data/real/',
        img_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    240,
                    240,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    240,
                    240,
                ), type='RandomCrop'),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                degree=(
                    -90,
                    90,
                ),
                flip_prob=0.5,
                rotate_prob=0.75,
                type='RandomRotFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='GlottisDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            240,
            240,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        240,
        240,
    ), type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(
        degree=(
            -90,
            90,
        ),
        flip_prob=0.5,
        rotate_prob=0.75,
        type='RandomRotFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.5, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.75, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path='images_filtered_20%_no_cords_backup/validation',
            seg_map_path='masks_adjusted_backup_8bit_2color/gtFine/Train'),
        data_root='data/real/',
        img_suffix='.png',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(scale=(
                240,
                240,
            ), type='Resize'),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        seg_map_suffix='.png',
        type='GlottisDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/ddrnet_23-slim_in1k-pre_2xb6-120k_glottis'
