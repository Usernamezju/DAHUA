"""NTU120 response distillation on NTU25 bone inputs."""

import os
import sys
from pathlib import Path


repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))
from dahua_cup.semantic_teacher.distillation import protogcn_adapter  # noqa: F401,E402


modality = "b"
graph = "nturgb+d"
work_dir = os.environ.get(
    "DAHUA_NTU120_DISTILL_WORK_DIR",
    "/workspace/data/xzz_data/DAHUA/runtime/visualization/evolution/training",
)
ann_file = os.environ.get(
    "DAHUA_NTU120_DISTILL_ANN",
    "/workspace/data/xzz_data/DAHUA/runtime/visualization/evolution/current.pkl",
)
load_from = os.environ.get("DAHUA_PROTOGCN_NTU120_INIT") or None

model = dict(
    type="DistillRecognizerGCN",
    backbone=dict(
        type="ProtoGCN",
        num_prototype=100,
        tcn_ms_cfg=[
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            ("max", 3),
            "1x1",
        ],
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
        num_classes=120,
        in_channels=384,
        weight=0.3,
    ),
    distill_cfg=dict(
        temperature=2.0,
        hard_weight=1.0,
        pseudo_weight=0.5,
        knowledge_weight=1.0,
        previous_weight=0.0,
        csc_weight=0.2,
    ),
)

dataset_type = "PoseDataset"
distill_keys = [
    "keypoint",
    "label",
    "has_hard_label",
    "teacher_distribution",
    "teacher_valid",
    "quality_weight",
    "previous_distribution",
    "previous_valid",
]
distill_tensor_keys = [
    "keypoint",
    "has_hard_label",
    "teacher_distribution",
    "teacher_valid",
    "quality_weight",
    "previous_distribution",
    "previous_valid",
]
train_pipeline = [
    dict(type="PreNormalize3D", align_spine=False),
    dict(type="RandomRot", theta=0.15),
    dict(type="Spatial_Flip", dataset=graph, p=0.5),
    dict(type="GenSkeFeat", dataset=graph, feats=[modality]),
    dict(type="UniformSampleDecode", clip_len=100),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="Collect", keys=distill_keys, meta_keys=[]),
    dict(type="ToTensor", keys=distill_tensor_keys),
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
    videos_per_gpu=int(os.environ.get("DAHUA_NTU120_DISTILL_BATCH", "8")),
    workers_per_gpu=int(os.environ.get("DAHUA_NTU120_DISTILL_WORKERS", "4")),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline,
        split="xsub_train",
    ),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split="xsub_val",
    ),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split="xsub_val",
    ),
)
optimizer = dict(
    type="SGD",
    lr=float(os.environ.get("DAHUA_NTU120_DISTILL_LR", "0.0005")),
    momentum=0.9,
    weight_decay=0.0005,
    nesterov=True,
)
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
lr_config = dict(policy="CosineAnnealing", min_lr=1e-6, by_epoch=False)
total_epochs = int(os.environ.get("DAHUA_NTU120_DISTILL_EPOCHS", "10"))
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
auto_resume = False
