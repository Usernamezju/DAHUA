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
docs/                 design, operations and acceptance material
runtime/              local runtime data (ignored by Git)
scripts/              shell launchers only
```

This is a standard Python `src` layout: `src/dahua_cup/` is the importable
application package. Data construction, offline evaluation and maintenance
commands are installed from that package; `scripts/` contains only the Web
shell launcher. There are no compatibility symlinks or duplicate projects.

## Install

Create the environment matching your target platform, then install this
project in editable mode. The runtime stack needs the PyTorch/MMCV versions
compatible with the bundled ProtoGCN code.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

For the optional Qwen teacher, additionally install `.[teacher]` in its
dedicated environment and set `DAHUA_QWEN_MODEL_DIR`.

## Web UI (server)

Run the Web service from the repository deployed on the server:

```bash
cd /workspace/code/DAHUA
bash scripts/run_visualization.sh
```

The launcher activates the server's `skel_gcn38` Conda environment
automatically when it is available; otherwise it falls back to `.venv` in the
repository.  To choose a different environment, set
`DAHUA_VIS_VENV=/path/to/venv`.  The service listens on `0.0.0.0:8000` by
default; set `DAHUA_VIS_PORT` to use another port.

The Web process, RTMPose extraction, student inference, and Qwen teacher can
use different server environments.  Select the built-in environment names
(`skel_gcn38`, `rtmpose26`, or `llm_env`) per role when launching; changes take
effect after restarting the service:

```bash
./scripts/run_visualization.sh \
  --web-env skel_gcn38 \
  --rtmpose-env rtmpose26 \
  --student-env skel_gcn38 \
  --teacher-env llm_env
```

## Run

The production Web student loads the accepted M1KD QAT INT8 checkpoint through
the portable quantized-state loader. The current PyTorch path restores those
weights into the CUDA inference graph; native INT8 kernel acceleration requires
a TensorRT, RKNN, or equivalent backend and is not claimed here.

```bash
export DAHUA_CODE_ROOT="$PWD"
export DAHUA_DATA_ROOT="$PWD/runtime"
export DAHUA_CAMPUS6_CONFIG="$PWD/third_party/ProtoGCN/configs/campus6/rtmpose26_k400_2d_gap_full.py"
export DAHUA_CAMPUS6_CHECKPOINT="$PWD/models/student/campus6_protogcn_gap_fp32_epoch40.pth"

python -m dahua_cup.pipeline.rtmpose17_pose_worker --help
python -m dahua_cup.pipeline.rtmpose17_student_worker --help
python -m dahua_cup.semantic_teacher.distillation.campus6_compression --help
dahua-build-annotations --help
dahua-audit-pose --help
dahua-cache-int8 --help
dahua-analyze-hard-samples --help
dahua-verify
```

Operational requirements, data construction, teacher routing, model promotion,
and acceptance limitations are documented under `docs/`. Datasets, Qwen model
files, credentials, and experiment outputs are intentionally not bundled.
