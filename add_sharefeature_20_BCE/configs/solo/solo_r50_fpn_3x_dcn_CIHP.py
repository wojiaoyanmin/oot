_base_=[
    '../_base_/models/solo_r50_fpn_dcn.py',
    '../_base_/datasets/CIHP_humanparsing.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
    ]
model=dict(
    ins_head=dict(
        num_classes=19)#不算background  但是多了一类“human”
)
data = dict(
    samples_per_gpu=3)
optimizer = dict(type='SGD', lr=0.006, momentum=0.9, weight_decay=0.0001)
#optimizer_config = dict(grad_clip=dict(max_norm=35,norm_type=2))
total_epochs = 72
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[54,65])
