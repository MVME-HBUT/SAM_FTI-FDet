_base_ = ['_base_/sam_fti-fdet.py']
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=10, save_best='coco/segm_mAP', rule='greater', save_last=True),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=False, interval=1, test_out_dir='vis_data')
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

num_classes = 15
prompt_shape = (10, 4)  # (per img pointset, per pointset point)

#### should be changed when using different pretrain model

hf_sam_pretrain_name = "pretrain/sam_cache/sam-vit-base"
# huggingface model name, e.g. facebook/sam-vit-base
hf_sam_pretrain_ckpt_path = "pretrain/sam_cache/sam-vit-base/sam_hq_vit_tiny.pth"

model = dict(
    decoder_freeze=False,
    shared_image_embedding=dict(
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path),
    ),
    backbone=dict(
        hf_pretrain_name=hf_sam_pretrain_name,
        init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)),
    neck=dict(
        feature_aggregator=dict(
            in_channels=hf_sam_pretrain_name,
            hidden_channels=32,
            select_layers=[1, 2],
        ),
    ),
    panoptic_head=dict(
        decoder_plus=True,
        mask_decoder=dict(
            hf_pretrain_name=hf_sam_pretrain_name,
            init_cfg=dict(type='Pretrained', checkpoint=hf_sam_pretrain_ckpt_path)
        ),
        per_pointset_point=prompt_shape[1],
        with_sincos=True,
        num_things_classes=num_classes,
        num_queries=prompt_shape[0],
        loss_cls=dict(
            class_weight=[1.0] * num_classes + [0.1])
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_classes
    ),
    test_cfg=dict(
        max_per_image=prompt_shape[0],
    )
)

dataset_type = 'CocoDataset'
data_root = '/data/coco/'

batch_size_per_gpu = 2
num_workers = 8
persistent_workers = True
train_dataloader = dict(
    batch_size=batch_size_per_gpu,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
    )
)

val_dataloader = dict(
    batch_size=batch_size_per_gpu*4,
    num_workers=num_workers,
    persistent_workers=persistent_workers,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
    )
)

test_dataloader = val_dataloader
resume = False
load_from = None

base_lr = 0.0001
max_epochs = 200

train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.001,
        begin=1,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True
    )
]

#### DeepSpeed Configs
runner_type = 'FlexibleRunner'
strategy = dict(
    type='DeepSpeedStrategy',
    fp16=dict(
        enabled=True,
        auto_cast=False,
        fp16_master_weights_and_grads=False,
        loss_scale=0,
        loss_scale_window=500,
        hysteresis=2,
        min_loss_scale=1,
        initial_scale_power=15,
    ),
    gradient_clipping=0.1,
    inputs_to_half=['inputs'],
    zero_optimization=dict(
        stage=2,
        allgather_partitions=True,
        allgather_bucket_size=2e8,
        reduce_scatter=True,
        reduce_bucket_size='auto',
        overlap_comm=True,
        contiguous_gradients=True,
    ),
)
optim_wrapper = dict(
    type='DeepSpeedOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=base_lr,
        weight_decay=0.05
    )
)

# #### AMP training config
# runner_type = 'Runner'
# optim_wrapper = dict(
#     type='AmpOptimWrapper',
#     dtype='float16',
#     optimizer=dict(
#         type='AdamW',
#         lr=base_lr,
#         weight_decay=0.05)
# )