# Campus6 delivery layout

`src/dahua_cup` is one importable Python package.  It contains only the
Campus6 runtime, training/quantization loop, review API and its static UI.

- `pipeline/`: RTMDet/RTMPose extraction, ProtoGCN inference, training and review-loop commands.
- `evaluation/`: acceptance metrics, pose-quality audit and existing-skeleton INT8 cache generation.
- `semantic_teacher/`: optional, externally configured Qwen teacher contract and distillation/QAT utilities; no Qwen weights are bundled.
- `configs/`: Campus6 labels and routing thresholds.
- `scripts/`: the single Web shell launcher; Python commands are package entry points from `pyproject.toml`.
- `third_party/ProtoGCN/`: pinned model implementation, kept separate from application code.
- `models/`: ignored binary weights with a tracked manifest.
- `tests/`: unit and integration regression tests; not imported by the running Web service.
- `runtime/`: ignored local artifacts, exports, review database, settings and videos.

The obsolete top-level `experiments/` matrix and its old M1FKD benchmark report
were removed. Reproducible Campus6 training, QAT/KD, evaluation and release code
now lives under the importable package.

The historical NTU120, MediaPipe, NanoDet, STN and automatic-evolution paths
are intentionally absent from this delivery repository.
