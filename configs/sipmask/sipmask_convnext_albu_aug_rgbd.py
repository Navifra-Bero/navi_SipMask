# model settings
custom_imports = dict(imports=['mmcls.models'], allow_failed_imports=False)
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-small_3rdparty_32xb128-noema_in1k_20220301-303e75e3.pth'  # noqa


model = dict(
    type='SipMask',
    backbone=dict(
        type='mmdet.models.backbones.mobilenet',
        arch='small',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.6,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        # in_channels=4,  # 4채널 입력으로 설정
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    neck=[
        dict(
            type='FPN',
            in_channels=[96, 192, 384, 768],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(type='DyHead', in_channels=256, out_channels=256, num_blocks=6)
    ],
    bbox_head=dict(
        type='SipMaskHead',
        num_classes=5,
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
        center_sample_radius=1.5))

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
    score_thr=0.1,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=200)
# dataset settings
dataset_type = 'RGBDCocoDataset'
classes = ('icebox', 'box', 'pouch', 'sack', 'unknown')
img_norm_cfg = dict(
    mean=[123.68, 116.78, 103.94], std=[58.40, 57.12, 57.38], to_rgb=True)

albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[-0.5, 0.5],
        contrast_limit=[-0.5, 0.5],
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1.0),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=1.0)
        ],
        p=0.1),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.1),
]

rgbd_trans = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.0625,
        scale_limit=0.0,
        rotate_limit=0,
        interpolation=1,
        p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Affine', shear={'x': (-20,20)}, p=1.0),
            dict(type='Affine', shear={'y': (-20,20)}, p=1.0),
        ],
        p=0.2),
]

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
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img_rgb': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='DefaultFormatBundle'),
    dict(type='ConcatRGBD', keys=['img_rgb', 'img_depth'], output_key='img'),  # RGB와 Depth를 결합하여 4채널로 만듦
    dict(
        type='Albu',
        transforms=rgbd_trans,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
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
#ensenso
data_root = '/home/rise/Documents/pkb/'
# data_root = '/home/rise/Desktop/graduation/under_segm/'
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train_RGB.json',
        classes=classes,
        img_prefix_rgb=data_root + 'image/RGB/',  # RGB 이미지 경로
        img_prefix_depth=data_root + 'image/depths/',  # Depth 이미지 경로
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_RGB.json',
        classes=classes,
        img_prefix_rgb=data_root + 'image/RGB/',  # RGB 이미지 경로
        img_prefix_depth=data_root + 'image/depths/',  # Depth 이미지 경로
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_RGB.json',
        classes=classes,
        img_prefix_rgb=data_root + 'image/RGB/',  # RGB 이미지 경로
        img_prefix_depth=data_root + 'image/depths/',  # Depth 이미지 경로
        pipeline=test_pipeline))


#kinect
# data_root = '/home/rise/Desktop/graduation/CalibNet/datasets/CUSTOM_kinect/'

# data = dict(
#     imgs_per_gpu=1,
#     workers_per_gpu=1,
#     train=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/train_RGB.json',
#         classes=classes,
#         img_prefix_rgb=data_root + 'image/RGB/',  # RGB 이미지 경로
#         img_prefix_depth=data_root + 'image/depths/',  # Depth 이미지 경로
#         pipeline=train_pipeline),
#     val=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/val_RGB.json',
#         classes=classes,
#         img_prefix_rgb=data_root + 'image/RGB/',  # RGB 이미지 경로
#         img_prefix_depth=data_root + 'image/depths/',  # Depth 이미지 경로
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         ann_file=data_root + 'annotations/val_RGB.json',
#         classes=classes,
#         img_prefix_rgb=data_root + 'image/RGB/',  # RGB 이미지 경로
#         img_prefix_depth=data_root + 'image/depths/',  # Depth 이미지 경로
#         pipeline=test_pipeline))


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
work_dir = './work_dirs/rgbd_conv_albu/'
load_from = None
resume_from = None
workflow = [('train', 1)]
