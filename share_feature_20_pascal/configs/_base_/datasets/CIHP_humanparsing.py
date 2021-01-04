dataset_type = 'CIHPDataset'
data_root = 'data/CIHP/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=False,with_instance=True),
    dict(type='Resize',
        img_scale=[(1333, 544), (1333, 608), (1333, 672), (1333, 736),
                   (1333, 800), (1333, 864)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip',flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=64),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes','gt_labels','gt_masks','gt_instances']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333,800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=64),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=10,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root +
                             'annotations/Instance_train.json',
            img_prefix=data_root + 'train/',
            seg_prefix=data_root + 'train/Category_ids/',
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
                          'annotations/Instance_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
                          'annotations/Instance_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
evaluation = dict(metric=['segm'])
