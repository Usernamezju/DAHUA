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
python -m pip install -r requirements/local.txt
python -m pytest
```

For the optional Qwen teacher, additionally install `.[teacher]` in its
dedicated environment and set `DAHUA_QWEN_MODEL_DIR`.

The remote Qwen host installs its own stack from `requirements/server.txt`.
Configure only its SSH host, port, user, and project root in **系统设置**; its
Python interpreter, CUDA setup, and Qwen model path remain server-side.

## Web UI (server)

Run the Web service from the repository deployed on the server:

```bash
cd /workspace/code/DAHUA_delivery
bash scripts/run_visualization.sh
```

重复执行该命令时，如果指定端口已有 Dahua Web 服务，脚本会保留现有服务并正常退出，不会再报端口占用。需要加载新代码时使用：

```bash
bash scripts/run_visualization.sh --restart
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
export DAHUA_CAMPUS6_DEPLOY_CONFIG="$PWD/third_party/ProtoGCN/configs/campus6/rtmpose26_k400_2d_gap_full.py"
export DAHUA_CAMPUS6_DEPLOYMENT_CHECKPOINT="$PWD/models/student/M1KD.int8.pt"

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

## 11. 项目代码结构与关键接口

生产入口是 `dahua_cup.backend.app`，由 `scripts/run_visualization.sh` 启动。
服务器部署目录为 `/workspace/code/DAHUA_delivery`；运行数据、SQLite 审核库、
产物与日志由 `DAHUA_DATA_ROOT`、`DAHUA_VIS_RUNTIME_ROOT` 和
`DAHUA_VIS_ARTIFACT_ROOT` 指向，均不应提交到 Git。

```text
DAHUA/
├── src/dahua_cup/              主应用包
│   ├── backend/                FastAPI、任务和审核库
│   ├── visualization/          Web 前端
│   ├── pipeline/               推理、训练和离线工具
│   ├── semantic_teacher/       Qwen 教师、伪标签和蒸馏
│   ├── feature_extraction/     骨架特征摘要
│   ├── evaluation/             评估与审计
│   ├── configs/campus/         Campus6 配置
│   └── demo/                   本地演示（非生产入口）
├── scripts/                    启动脚本
├── third_party/ProtoGCN/       固定的上游模型代码与配置
├── tests/                      单元与集成测试
├── docs/                       设计、部署与验收文档
├── models/                     本地模型；仅跟踪 MANIFEST
└── runtime/                    运行数据、审核库、产物和日志（不提交）
```

### 核心数据流

```text
Campus6 既有 COCO-17 / 上传视频
  → RTMPose 骨架特征与匿名骨架视频
  → M1KD INT8 六类概率
  → Hard(x) = Top-1 ≤ τ ∧ (C1 ∨ C2 ∨ C3 ∨ C5)
  → Qwen 教师（仅难例）→ 人工审核 → 导出与增量训练
```

`C1`–`C5` 分别覆盖学生不确定、师生不一致、高置信冲突和低频类别。Qwen 不接收
学生概率，且教师结果必须包含结构化证据。

### FastAPI 接口

| 接口 | 作用 |
| --- | --- |
| `GET /api/health` | 服务存活检查。 |
| `GET /api/capabilities`、`/api/system`、`/api/gpus` | 系统、模型和 GPU 状态。 |
| `GET /api/dashboard`、`/api/models/status`、`/api/training/status` | 运行、模型和训练状态。 |
| `GET/PUT /api/macro-parameters` | 读取或更新难例门控参数。 |
| `GET /api/samples`、`/api/samples/{sample_id}`、`/api/samples/{sample_id}/prediction` | 样本、预测和审核详情。 |
| `GET /api/samples/{sample_id}/media/pose` | 匿名骨架视频。 |
| `GET /api/hard-samples` | 待审核难例队列。 |
| `POST /api/samples/{sample_id}/jobs`、`GET /api/jobs/{job_id}` | 提交或查询处理任务。 |
| `POST /api/samples/{sample_id}/review`、`POST /api/samples/{sample_id}/undo`、`POST /api/reviews/undo-last` | 提交或撤销审核。 |
| `POST /api/samples/import-manifest`、`POST /api/datasets/export` | 导入样本或导出审核数据。 |

未知的非 `/api` 路径由 `visualization/` 静态页面处理；前端不接收原始 RGB 视频，
只请求服务器派生的骨架媒体。
