"""COCO-style calibration loader for RTMPose-S TensorRT INT8 export.

Set ``DAHUA_POSE_INT8_CALIB_DIR`` to a directory containing extracted RGB
frames and its generated ``annotations.json`` (with per-image person bboxes
produced by the INT8 detector, so TopDownAffine crops match runtime).
It intentionally inherits the shipped RTMPose test transforms, so calibration
preprocessing matches export.
"""

import os


_base_ = (
    "/workspace/code/envs/rtmpose_int8/lib/python3.8/site-packages/"
    "mmpose/.mim/configs/body_2d_keypoint/rtmpose/coco/"
    "rtmpose-s_8xb256-420e_coco-256x192.py"
)

_calibration_root = os.environ["DAHUA_POSE_INT8_CALIB_DIR"]

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        data_root=_calibration_root,
        ann_file="annotations.json",
        data_prefix=dict(img=""),
    ),
)
