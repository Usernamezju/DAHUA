# System-design alignment

This package was extracted against the two Notion pages named “系统设计文档”.
The active delivery boundary is Campus6 rather than the unrelated NTU120/ZSL
research line.

- Input and feature contract: RTMDet/RTMPose, at most two temporally tracked
  COCO-17 people, normalized 2D coordinates, confidence mask, and 100-frame
  ProtoGCN processing.
- Student: Kinetics-400-pretrained ProtoGCN, Campus6 six-class fine-tuning,
  with GAP semantic supervision used during training.
- Teacher and safety: closed-set Qwen output, low confidence/margin/quality or
  instability routing, structured validation, and human review on conflicts.
- Learning loop: pseudo-label filtering, replay, distillation, candidate
  validation and release gates are retained with their tests.
- Deployment: M1FKD portable INT8 artifact is bundled as the edge-default;
  FP32 GAP weights and the QAT/KD/export/benchmark code are retained for
  reproducibility. The current portable loader dequantizes before PyTorch
  execution, so native INT8 acceleration is explicitly not asserted.

Excluded from this repository: independent `our_model` ZSL/GZSL experiments,
unrelated GCN baselines, their datasets, and research-only checkpoints.
