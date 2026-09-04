"""Campus6 COCO-17 ProtoGCN with GAP under the RTMDet-S/RTMPose-S contract."""

import os
from pathlib import Path

_base_ = ["./rtm_s_coco17_k400_2d_full.py"]

work_dir = os.environ.get(
    "DAHUA_CAMPUS6_WORK_DIR",
    "/workspace/data/xzz_data/DAHUA/experiments/ProtoGCN/campus6_rtm_s_coco17_k400_2d_gap_full",
)
_repository_root = Path(os.environ.get("DAHUA_CODE_ROOT", ".")).resolve()
_bundled_semantic_text = _repository_root / "models/semantic/campus6_gap_clip_v3.npy"

model = dict(
    type="RecognizerGCNGAP",
    semantic_text_path=os.environ.get("DAHUA_CAMPUS6_GAP_TEXT", str(_bundled_semantic_text)),
    semantic_loss_weight=.15,
    semantic_temperature=.07,
)
