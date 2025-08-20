# model settings
model = dict(
    type='SipMask',
    backbone=dict(
        type='ResnetAtt',  # Attention 기반의 ResNet 백본을 사용
        rgb_backbone_cfg=dict(
            depth=50,  # ResNet의 깊이 설정 (예: 50)
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            in_channels=3,  # RGB는 3채널 입력
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
            style='pytorch'
        ),
        depth_backbone_cfg=dict(
            depth=50,  # ResNet의 깊이 설정 (예: 50)
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            in_channels=1,  # Depth는 1채널 입력
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
            style='pytorch'
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],  # FPN에 전달되는 피처맵 채널
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        upsample_cfg=dict(mode='bilinear')),
    bbox_head=dict(
        type='SipMaskHead',
        num_classes=5,  # 분류할 클래스 수
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        center_sampling=True,
        center_sample_radius=1.5)
)

# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.45,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=200)
# dataset settings
dataset_type = 'RGBDCocoDataset'
classes = ('icebox', 'box', 'pouch', 'sack', 'unknown')
img_norm_cfg = dict(
    mean=[123.68, 116.78, 103.94], std=[58.40, 57.12, 57.38], to_rgb=True)

train_pipeline = [
    dict(type='LoadRGBDImages', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PhotoMetricDistortionRGB',
         brightness_delta=32,
         contrast_range=(0.5, 1.5),
         saturation_range=(0.5, 1.5),
         hue_delta=18),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='NormalizeDepth'),  # Depth 이미지 정규화
    dict(
        type='NormalizeRGBD',
        mean_rgb=[123.68, 116.78, 103.94],
        std_rgb=[58.4, 57.12, 57.38],
        mean_depth=[127.5],
        std_depth=[127.5],
        to_rgb=True
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='ConcatRGBD', keys=['img_rgb', 'img_depth'], output_key='img'),  # RGB와 Depth를 결합하여 4채널로 만듦
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])  # img는 이제 4채널 이미지
]

test_pipeline = [
    dict(type='LoadRGBDImages'),  # RGB와 Depth 각각 로드
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),
            # dict(type='NormalizeDepth'),  # Depth 이미지 정규화
            dict(
                type='NormalizeRGBD',
                mean_rgb=[123.68, 116.78, 103.94],
                std_rgb=[58.4, 57.12, 57.38],
                mean_depth=[127.5],
                std_depth=[127.5],
                to_rgb=True
            ),
            dict(type='Pad', size_divisor=32),
            dict(type='ConcatRGBD', keys=['img_rgb', 'img_depth'], output_key='img'),  # RGB와 Depth를 결합하여 4채널로 만듦
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data_root = '/home/rise/Documents/pkb/'
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_RGB.json',
        classes=classes,
        img_prefix_rgb=data_root + 'images/RGB/',  # RGB 이미지 경로
        img_prefix_depth=data_root + 'images/depths/',  # Depth 이미지 경로
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_RGB.json',
        classes=classes,
        img_prefix_rgb=data_root + 'images/RGB/',  # RGB 이미지 경로
        img_prefix_depth=data_root + 'images/depths/',  # Depth 이미지 경로
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_RGB.json',
        classes=classes,
        img_prefix_rgb=data_root + 'images/RGB/',  # RGB 이미지 경로
        img_prefix_depth=data_root + 'images/depths/',  # Depth 이미지 경로
        pipeline=test_pipeline))

evaluation = dict(interval=5, metric='segm')

# optimizer
optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0002,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'layer_wise',
        'num_layers': 16
    })

optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
      policy='CosineAnnealing',
      warmup='linear',
      warmup_iters=1000,
      warmup_ratio=1.0 / 10,
      min_lr_ratio=1e-5)

checkpoint_config = dict(interval=5)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
total_epochs = 150
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/sipmask_rgbd50_attention/'
load_from = None
resume_from = None
workflow = [('train', 1)]
