# Dahua Campus6 behavior recognition

The deployable Campus6 project, extracted from the mixed research workspace.
It implements the documented six-class pipeline:

```text
video -> RTMDet/RTMPose (two COCO-17 tracks) -> ProtoGCN student
      -> optional Qwen teacher -> review / pseudo-label / retraining loop
```

The default student artifact is `models/student/M1KD.int8.pt`.
It is the accepted QAT INT8 model trained with logits knowledge distillation
(Val 47/53, Test 46/53, All 345/358). The 42.50 MB FP32 GAP
checkpoint remains in `models/student/` as the training teacher and regression
baseline. See `models/MANIFEST.json` for the weight hash and acceptance metrics.

## Layout

```text
src/dahua_cup/       application, training, inference, Web API and tests' source
third_party/ProtoGCN/ pinned upstream model implementation and Campus6 configs
tests/                unit and integration tests
models/               local weights (ignored by Git; manifest is tracked)
reports/              checked benchmark and deployment reports
docs/                 design, operations and acceptance material
```

This is a standard Python `src` layout: `src/dahua_cup/` is the importable
application package, while `scripts/` holds only command-line maintenance
tools. There are no compatibility symlinks or nested duplicate projects.

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

The production Web student loads the accepted M1KD QAT INT8 checkpoint through
the portable quantized-state loader. The current PyTorch path restores those
weights into the CUDA inference graph; native INT8 kernel acceleration requires
a TensorRT, RKNN, or equivalent backend and is not claimed here.

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
