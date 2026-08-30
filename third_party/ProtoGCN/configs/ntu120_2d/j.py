import os

modality = 'j'
graph = 'coco_new'
work_dir = os.environ.get('DAHUA_NTU120_WORK_DIR', '/workspace/data/xzz_data/DAHUA/experiments/pipeline/protogcn_ntu120_2d')
videos_per_gpu = int(os.environ.get('DAHUA_PROTOGCN_BATCH_PER_GPU', '32'))
workers_per_gpu = int(os.environ.get('DAHUA_PROTOGCN_WORKERS_PER_GPU', '8'))
learning_rate = float(os.environ.get('DAHUA_PROTOGCN_LR', '0.0125'))
training_epochs = int(os.environ.get('DAHUA_PROTOGCN_EPOCHS', '150'))

model = dict(
    type='RecognizerGCN',
    backbone=dict(type='ProtoGCN', num_prototype=400,
                  tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                  graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02)),
    cls_head=dict(type='SimpleHead', joint_cfg=graph, num_classes=120, in_channels=384, weight=0.2))

dataset_type = 'PoseDataset'
ann_file = os.environ.get('DAHUA_NTU120_2D_ANN', '/workspace/data/xzz_data/DAHUA/datasets/ntu120_2d/ntu120_2d.pkl')
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'), dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='Kinetics_Transform', dataset=graph), dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    dict(type='FormatGCNInput', num_person=2), dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1),
    dict(type='PoseDecode'), dict(type='Kinetics_Transform', dataset=graph),
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]), dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]), dict(type='ToTensor', keys=['keypoint'])]
test_pipeline = [dict(item) for item in val_pipeline]

data = dict(
    videos_per_gpu=videos_per_gpu, workers_per_gpu=workers_per_gpu,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=ann_file, split='xsub_train', pipeline=train_pipeline),
    val=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', pipeline=test_pipeline))
optimizer = dict(type='SGD', lr=learning_rate, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
find_unused_parameters = False
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = training_epochs
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
