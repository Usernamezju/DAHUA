"""Campus6 Qwen/previous-model response distillation on NTU25 bone inputs."""

from pathlib import Path
import sys


_base_ = ["./campus6_ntu25_bone.py"]
repository_root = Path(__file__).resolve().parents[3]
if str(repository_root) not in sys.path:
    sys.path.insert(0, str(repository_root))
from dahua_cup.semantic_teacher.distillation import protogcn_adapter  # noqa: F401,E402


model = dict(
    type="DistillRecognizerGCN",
    distill_cfg=dict(
        temperature=2.0,
        hard_weight=1.0,
        pseudo_weight=0.5,
        knowledge_weight=1.0,
        previous_weight=0.5,
        csc_weight=0.3,
    ),
)

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
    dict(type="Spatial_Flip", dataset="nturgb+d", p=0.5),
    dict(type="GenSkeFeat", dataset="nturgb+d", feats=["b"]),
    dict(type="UniformSampleDecode", clip_len=100),
    dict(type="FormatGCNInput", num_person=2),
    dict(type="Collect", keys=distill_keys, meta_keys=[]),
    dict(type="ToTensor", keys=distill_tensor_keys),
]
data = dict(train=dict(pipeline=train_pipeline))
