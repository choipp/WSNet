_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/cityscapes.py',
    '../_base_/schedules/schedule_160k.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='EncoderDecoder',
    pretrained='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth',
    backbone=dict(
        type='pvt_v2_b2',
        style='pytorch'),
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(num_classes=150))

# optimizer
optimizer = dict(_delete_= True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)