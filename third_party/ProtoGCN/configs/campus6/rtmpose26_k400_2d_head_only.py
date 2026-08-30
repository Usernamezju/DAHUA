"""Campus6 Halpe-26 -> COCO-17 -> Kinetics-2D ProtoGCN, frozen backbone."""

import os

modality = "j"
graph = "coco_new"
freeze_backbone = True
work_dir = os.environ.get(
    "DAHUA_CAMPUS6_WORK_DIR",
    "/workspace/data/xzz_data/DAHUA/experiments/ProtoGCN/campus6_rtmpose26_k400_2d_head_only",
)
load_from = os.environ.get(
    "DAHUA_CAMPUS6_PRETRAINED",
    "/workspace/data/xzz_data/DAHUA/models/protogcn_pretrained/kinetics_skeleton_2d_joint/k400_j1_backbone_only.pth",
)
ann_file = os.environ.get(
    "DAHUA_CAMPUS6_ANN",
    "/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1/rtmpose26_coco17/annotations_protogcn_2d.pkl",
)

model = dict(
    type="RecognizerGCN",
    freeze_backbone=True,
    backbone=dict(
        type="ProtoGCN", num_prototype=400,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"],
        graph_cfg=dict(layout=graph, mode="random", num_filter=8, init_off=.04, init_std=.02),
    ),
    cls_head=dict(type="SimpleHead", joint_cfg=graph, num_classes=6, in_channels=384, weight=.2),
)

dataset_type = "PoseDataset"
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type="UniformSampleFrames", clip_len=100), dict(type="PoseDecode"),
    dict(type="Flip", flip_ratio=.5, left_kp=left_kp, right_kp=right_kp),
    dict(type="Kinetics_Transform", dataset=graph),
    dict(type="GenSkeFeat", dataset=graph, feats=[modality]),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]
eval_pipeline = [
    dict(type="UniformSampleFrames", clip_len=100, num_clips=1), dict(type="PoseDecode"),
    dict(type="Kinetics_Transform", dataset=graph),
    dict(type="GenSkeFeat", dataset=graph, feats=[modality]),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="Collect", keys=["keypoint", "label"], meta_keys=[]),
    dict(type="ToTensor", keys=["keypoint"]),
]
data = dict(
    videos_per_gpu=24, workers_per_gpu=4, test_dataloader=dict(videos_per_gpu=8),
    train=dict(type=dataset_type, ann_file=ann_file, split="train", pipeline=train_pipeline,
               class_prob=[1, 1, 1, 1, 5, 2]),
    val=dict(type=dataset_type, ann_file=ann_file, split="val", pipeline=eval_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split="test", pipeline=eval_pipeline),
)
optimizer = dict(type="SGD", lr=.02, momentum=.9, weight_decay=.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy="CosineAnnealing", min_lr=0, by_epoch=False)
total_epochs = 30
checkpoint_config = dict(interval=5)
evaluation = dict(interval=5, metrics=["top_k_accuracy", "mean_class_accuracy"], topk=(1, 5), save_best="top1_acc")
log_config = dict(interval=10, hooks=[dict(type="TextLoggerHook")])
dist_params = dict(backend="nccl")
workflow = [("train", 1)]
