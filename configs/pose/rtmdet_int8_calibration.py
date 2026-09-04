"""COCO-style calibration loader for RTMDet-S TensorRT INT8 export.

Set ``DAHUA_POSE_INT8_CALIB_DIR`` to a directory containing extracted RGB
frames and its generated ``annotations.json``.  It intentionally inherits the
shipped RTMDet test transforms, so calibration preprocessing matches export.
"""

import os


_base_ = (
    "/workspace/code/envs/rtmpose_int8/lib/python3.8/site-packages/"
    "mmdet/.mim/configs/rtmdet/rtmdet_s_8xb32-300e_coco.py"
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
