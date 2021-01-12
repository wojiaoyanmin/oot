model=dict(
    type='YMDetector',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0,1,2,3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(
            type='DCNv2',
            deformable_groups=1,
            fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5,
        add_extra_convs=True),#和下面的num_ints=4一起修改
    ins_head=dict(
        type='SOLOHead',
        num_classes=19,#不算background
        in_channels=256,
        stacked_convs=4,
        seg_feat_channels=512,
        #conv_cfg=dict(type='DCNv2'),
        strides=[8, 8, 16, 16],
        #scale_ranges=((1, 48), (24, 96), (48, 192), (96,2048)),
        scale_ranges=((1, 96), (48, 192), (96, 384), (192,2048)),
        sigma=0.2,
        num_human_grids=20,
        num_grids=[40, 36, 24, 16],
        ins_out_channels=256,
        loss_ins=dict(
            type='DiceLoss',
            loss_weight=3.0),
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        human_loss_ins=dict(
            type='DiceLoss',
            loss_weight=3.0),
        human_loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    mask_feat_head=dict(
        type='MaskFeatHead',
        in_channels=256,
        out_channels=128,
        start_level=0,
        end_level=4,
        num_classes=256,
        conv_cfg=dict(type='DCNv2'),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
)
train_cfg=dict()
test_cfg=dict(
    nms_pre=200,
    score_thr=1/3,
    human_score_thr=0.1,
    mask_thr=0.5,
    update_thr=0.1,#0.05,0.1
    human_update_thr=0.1,
    assign_parts_th=2/3,
    kernel='gaussian',
    sigma=2.0,
    max_per_img=200,
    mix_pixels=20)

