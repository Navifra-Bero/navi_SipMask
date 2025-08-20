# dataset settings
dataset_type = 'RGBDCocoDataset'
data_root = '/home/rise/Documents/pkb/'
classes = ('icebox', 'box', 'pouch', 'sack', 'unknown')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
        img_prefix_rgb=data_root + 'images/RGB_dark/',  # RGB 이미지 경로
        img_prefix_depth=data_root + 'images/depths/',  # Depth 이미지 경로
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_RGB.json',
        classes=classes,
        img_prefix_rgb=data_root + 'images/RGB_dark/',  # RGB 이미지 경로
        img_prefix_depth=data_root + 'images/depths/',  # Depth 이미지 경로
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_RGB.json',
        classes=classes,
        img_prefix_rgb=data_root + 'images/RGB_dark/',  # RGB 이미지 경로
        img_prefix_depth=data_root + 'images/depths/',  # Depth 이미지 경로
        pipeline=test_pipeline))
evaluation = dict(interval=5, metric='segm')
