"""Campus6 fine-tuning config for MediaPipe-derived NTU25 bone features."""

import os


modality = "b"
graph = "nturgb+d"
work_dir = os.environ.get(
    "DAHUA_CAMPUS6_WORK_DIR",
    "/workspace/data/xzz_data/DAHUA/experiments/pipeline/protogcn_campus6/finetune",
)
ann_file = os.environ.get(
    "DAHUA_CAMPUS6_ANN",
    "/workspace/data/xzz_data/DAHUA/datasets/campus6/current/annotations.pkl",
)
load_from = os.environ.get("DAHUA_PROTOGCN_CAMPUS6_INIT") or None

model = dict(
    type="RecognizerGCN",
    backbone=dict(
        type="ProtoGCN",
        num_prototype=100,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"],
        graph_cfg=dict(
            layout=graph,
            mode="random",
            num_filter=8,
            init_off=0.04,
            init_std=0.02,
        ),
    ),
    cls_head=dict(
        type="SimpleHead",
        joint_cfg=graph,
        num_classes=6,
        in_channels=384,
        weight=0.3,
        dropout=0.2,
    ),
)

dataset_type = "PoseDataset"
train_pipeline = [
    dict(type="PreNormalize3D", align_spine=False),
    dict(type="RandomRot", theta=0.15),
    dict(type="Spatial_Flip", dataset=graph, p=0.5),
    dict(type="GenSkeFeat", dataset=graph, feats=[modality]),
    dict(type="UniformSampleDecode", clip_len=100),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]
val_pipeline = [
    dict(type="PreNormalize3D", align_spine=False),
    dict(type="GenSkeFeat", dataset=graph, feats=[modality]),
    dict(type="UniformSampleDecode", clip_len=100, num_clips=1),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]
test_pipeline = [
    dict(type="PreNormalize3D", align_spine=False),
    dict(type="GenSkeFeat", dataset=graph, feats=[modality]),
    dict(type="UniformSampleDecode", clip_len=100, num_clips=5),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split="train"
    ),
    val=dict(
        type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split="val"
    ),
    test=dict(
        type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split="test"
    ),
)
optimizer = dict(
    type="SGD",
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
    paramwise_cfg=dict(custom_keys={"cls_head": dict(lr_mult=10.0)}),
)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy="CosineAnnealing", min_lr=1e-6, by_epoch=False)
total_epochs = 40
checkpoint_config = dict(interval=1)
evaluation = dict(
    interval=1,
    metrics=["top_k_accuracy", "mean_class_accuracy"],
    topk=(1, 5),
    save_best="mean_class_accuracy",
    rule="greater",
)
log_config = dict(interval=20, hooks=[dict(type="TextLoggerHook")])
find_unused_parameters = True
auto_resume = True
