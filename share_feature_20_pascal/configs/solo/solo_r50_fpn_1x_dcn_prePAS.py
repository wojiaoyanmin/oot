_base_=[
    '../_base_/models/solo_r50_fpn_dcn.py',
    '../_base_/datasets/PAS_humanparsing.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
    ]
model=dict(
    pretrained=None,
    ins_head=dict(
        num_classes=6)#不算background  但是多了一类“human”
)
load_from = 'work_dirs/sharefeature_20_dcn_1x/epoch_12.pth'
data = dict(
    samples_per_gpu=2)
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
total_epochs = 45
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    step=[34,41])