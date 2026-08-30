"""Leakage-safe Campus6 six-class ProtoGCN fine-tuning configuration."""

import os


modality = "j"
graph = "nturgb+d"
work_dir = os.environ.get(
    "DAHUA_CAMPUS6_WORK_DIR",
    "/workspace/data/xzz_data/DAHUA/experiments/ProtoGCN/campus6_leakage_safe_balanced_v1",
)
load_from = os.environ.get(
    "DAHUA_CAMPUS6_PRETRAINED",
    "/workspace/data/xzz_data/DAHUA/experiments/ProtoGCN/ntu120_xsub_joint/epoch_147.pth",
)
ann_file = os.environ.get(
    "DAHUA_CAMPUS6_ANN",
    "/workspace/data/xzz_data/DAHUA/datasets/campus6_mediapipe_ntu25/leakage_safe_v1/annotations_protogcn.pkl",
)

# Train counts are [189, 22, 40, 39, 4, 55].  The weights use scaled
# inverse-square-root frequency correction to avoid letting the four-shot
# class dominate each update.
class_prob = [1, 3, 2, 2, 8, 2]
class_weight = [0.359, 1.052, 0.780, 0.790, 2.468, 0.665]

model = dict(
    type="RecognizerGCN",
    backbone=dict(
        type="ProtoGCN",
        num_prototype=400,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"],
        graph_cfg=dict(layout=graph, mode="random", num_filter=8, init_off=.04, init_std=.02),
    ),
    cls_head=dict(
        type="SimpleHead", joint_cfg=graph, num_classes=6, in_channels=384,
        weight=0.0, dropout=0.25,
        loss_cls=dict(type="CrossEntropyLoss", class_weight=class_weight),
    ),
)

dataset_type = "PoseDataset"
train_pipeline = [
    dict(type="PreNormalize3D", align_spine=False),
    dict(type="RandomRot", theta=0.2),
    dict(type="Spatial_Flip", dataset="nturgb+d", p=0.5),
    dict(type="GenSkeFeat", feats=[modality]),
    dict(type="UniformSampleDecode", clip_len=100),
    dict(type="FormatGCNInput"),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]
eval_pipeline = [
    dict(type="PreNormalize3D", align_spine=False),
    dict(type="GenSkeFeat", feats=[modality]),
    dict(type="UniformSampleDecode", clip_len=100, num_clips=1),
    dict(type="FormatGCNInput"),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]

data = dict(
    videos_per_gpu=24,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=8),
    train=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline,
               split="train", class_prob=class_prob),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=eval_pipeline, split="val"),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=eval_pipeline, split="test"),
)

optimizer = dict(type="SGD", lr=0.003, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy="CosineAnnealing", min_lr=0, by_epoch=False)
total_epochs = 30
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metrics=["top_k_accuracy", "mean_class_accuracy"], topk=(1, 5))
log_config = dict(interval=10, hooks=[dict(type="TextLoggerHook")])
dist_params = dict(backend="nccl")
workflow = [("train", 1)]
