# Campus6 delivery layout

`src/dahua_cup` is one importable Python package.  It contains only the
Campus6 runtime, training/quantization loop, review API and its static UI.

- `pipeline/`: RTMDet/RTMPose extraction, ProtoGCN inference, training and review-loop commands.
- `semantic_teacher/`: optional, externally configured Qwen teacher contract and distillation/QAT utilities; no Qwen weights are bundled.
- `configs/`: Campus6 labels and routing thresholds.
- `scripts/`: four standalone data/checkpoint/Web launch helpers.
- `third_party/ProtoGCN/`: pinned model implementation, kept separate from application code.
- `models/`: ignored binary weights with a tracked manifest.

The historical NTU120, MediaPipe, NanoDet, STN and automatic-evolution paths
are intentionally absent from this delivery repository.
