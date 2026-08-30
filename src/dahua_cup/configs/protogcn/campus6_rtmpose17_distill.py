"""Campus6 direct COCO-17 Qwen-logit distillation configuration."""

from pathlib import Path
import os
import sys

_base_ = ["../../../gcn_models/ProtoGCN/configs/campus6/rtmpose26_k400_2d_full.py"]
# MMCV executes a temporary copy of this config, so ``__file__`` is not the
# repository location.  The launcher exports DAHUA_CODE_ROOT; cwd is the
# documented fallback for direct ``tools/dist_train.sh`` invocation.
repository_root = Path(os.environ.get("DAHUA_CODE_ROOT", Path.cwd())).resolve()
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))
from dahua_cup.semantic_teacher.distillation import protogcn_adapter  # noqa: F401,E402

model = dict(type="DistillRecognizerGCN", distill_cfg=dict(
    temperature=2.0, hard_weight=1.0, pseudo_weight=0.5,
    knowledge_weight=1.0, previous_weight=0.5, csc_weight=0.3,
))
distill_keys = ["keypoint", "label", "has_hard_label", "teacher_distribution", "teacher_valid", "quality_weight", "previous_distribution", "previous_valid"]
train_pipeline = [
    dict(type="UniformSampleFrames", clip_len=100), dict(type="PoseDecode"),
    dict(type="Kinetics_Transform", dataset="coco_new"),
    dict(type="GenSkeFeat", dataset="coco_new", feats=["j"]),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="Collect", keys=distill_keys, meta_keys=[]),
    dict(type="ToTensor", keys=[key for key in distill_keys if key != "label"]),
]
data = dict(train=dict(pipeline=train_pipeline))
