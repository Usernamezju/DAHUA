# Dahua Campus6 behavior recognition

The deployable Campus6 project, extracted from the mixed research workspace.
It implements the documented six-class pipeline:

```text
video -> RTMDet/RTMPose (two COCO-17 tracks) -> ProtoGCN student
      -> optional Qwen teacher -> review / pseudo-label / retraining loop
```

The default edge artifact is `models/student/M1FKD.deployment.int8.pt`.
It is the exported inference-only form of the QAT INT8 model trained with
both raw-logit and feature knowledge distillation. The 42.50 MB FP32 GAP
checkpoint remains in `models/student/` as the training teacher and regression
baseline. See `models/MANIFEST.json` and `reports/` for hashes and benchmark
evidence.

## Layout

```text
src/dahua_cup/       application, training, inference, Web API and tests' source
third_party/ProtoGCN/ pinned upstream model implementation and Campus6 configs
tests/                unit and integration tests
models/               local weights (ignored by Git; manifest is tracked)
reports/              checked benchmark and deployment reports
docs/                 design, operations and acceptance material
```

`dahua_cup -> src/dahua_cup` and `gcn_models -> third_party` are tracked
compatibility symlinks for legacy launchers. New code should import
`dahua_cup` and refer to `third_party/ProtoGCN` explicitly.

## Install

Create the environment matching your target platform, then install this
project in editable mode. The runtime stack needs the PyTorch/MMCV versions
compatible with the bundled ProtoGCN code.

```bash
python -m pip install -e '.[runtime,web,test]'
python -m pip install -r third_party/ProtoGCN/requirements.txt
pytest
```

For the optional Qwen teacher, additionally install `.[teacher]` in its
dedicated environment and set `DAHUA_QWEN_MODEL_DIR`.

## Run

The documented production path remains the FP32 Web student while native INT8
execution is not yet available in the current PyTorch loader. It may load the
portable INT8 checkpoint for validation, but dequantizes it before executing
FP32 CUDA operators. Do not claim native INT8 acceleration until a TensorRT,
RKNN, or equivalent backend is added and benchmarked.

```bash
export DAHUA_CODE_ROOT="$PWD"
export DAHUA_DATA_ROOT="$PWD/runtime-data"
export DAHUA_CAMPUS6_CONFIG="$PWD/third_party/ProtoGCN/configs/campus6/rtmpose26_k400_2d_gap_full.py"
export DAHUA_CAMPUS6_CHECKPOINT="$PWD/models/student/campus6_protogcn_gap_fp32_epoch40.pth"

python -m dahua_cup.pipeline.rtmpose17_pose_worker --help
python -m dahua_cup.pipeline.rtmpose17_student_worker --help
python -m dahua_cup.semantic_teacher.distillation.campus6_compression --help
dahua-verify
```

Operational requirements, data construction, teacher routing, model promotion,
and acceptance limitations are documented under `docs/`. Datasets, Qwen model
files, credentials, and experiment outputs are intentionally not bundled.
