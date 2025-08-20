# # optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)


# optimizer = dict(
#     constructor='LearningRateDecayOptimizerConstructor',
#     type='AdamW',
#     lr=0.0002,
#     betas=(0.9, 0.999),
#     weight_decay=0.05,
#     paramwise_cfg={
#         'decay_rate': 0.9,
#         'decay_type': 'layer_wise',
#         'num_layers': 16
#     })

# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#       policy='CosineAnnealing',
#       warmup='linear',
#       warmup_iters=1000,
#       warmup_ratio=1.0 / 10,
#       min_lr_ratio=1e-5)

