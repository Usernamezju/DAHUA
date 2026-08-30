"""Campus6 Kinetics-2D ProtoGCN full fine-tuning with training-only GAP loss."""

import os

_base_ = ["./rtmpose26_k400_2d_full.py"]

work_dir = os.environ.get("DAHUA_CAMPUS6_WORK_DIR", "/workspace/data/xzz_data/DAHUA/experiments/ProtoGCN/campus6_rtmpose26_k400_2d_gap_full_v2")
model = dict(
    type="RecognizerGCNGAP",
    semantic_text_path=os.environ.get("DAHUA_CAMPUS6_GAP_TEXT", "/workspace/data/xzz_data/AVA_Kinetics_competition_audit_v1/rtmpose26_coco17/campus6_gap_clip_v3.npy"),
    semantic_loss_weight=.15,
    semantic_temperature=.07,
)
