# Dahua Campus6 behavior recognition

The deployable Campus6 project, extracted from the mixed research workspace.
It implements the documented six-class pipeline:

```text
video -> RTMDet-nano-person/RTMPose-S (two COCO-17 tracks) -> ProtoGCN student
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

The same **系统设置** initialization also ensures the server-side GCN training
environment: when `DAHUA_INCREMENTAL_REMOTE_PYTHON` (default
`/root/miniconda3/envs/skel_gcn38/bin/python`) is missing on the remote host,
it is rebuilt automatically from `requirements/skel.txt` (Python 3.8 with the
PyTorch 1.10.2 / CUDA 11.3 Conda build). An existing environment is never
modified.

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

For a developer workstation, use the equivalent local launcher and supply the
three Conda interpreters explicitly.  It preflights FastAPI upload support,
RTMDet/RTMPose and PyTorch/MMCV before binding the Web port.  The local
launcher checks both worker environments: it uses GPU when both expose CUDA,
otherwise it automatically falls back to CPU. Qwen remains a conditional remote
call only after the student uncertainty gate fires:

```bash
bash scripts/run_local_visualization.sh \
  --web-python /path/to/web-env/bin/python \
  --rtmpose-python /path/to/pose-env/bin/python \
  --student-python /path/to/gcn-env/bin/python \
  --device auto
```

On this workstation the CPU-tested interpreters are
`/home/fjp/miniconda3/envs/dahua_cpu/bin/python` (Web + RTMDet/RTMPose) and
`/home/fjp/miniconda3/envs/dahua_gcn_cpu/bin/python` (ProtoGCN).  They are
kept separate because the former uses MMCV 2.x while the delivered ProtoGCN
uses MMCV 1.5.0. CPU works without a CUDA-visible GPU; a CUDA-capable pair of
environments is selected automatically when present.

Then open `http://127.0.0.1:8000/#gcn`.  Configure the remote Qwen SSH
connection once in **系统设置**; it is consulted only after the local GCN gate
asks for semantic review.

### Qwen API resilience fallback

The preferred teacher remains the configured server Qwen model.  If its SSH
connection, GPU admission, or inference fails, the Web service can
automatically fall back to a hosted Qwen OpenAI-compatible API.  This is an
optional resilience path, not a replacement for the server model.  It sends
only derived skeleton-video JPEG frames and the measured semantic-graph prompt;
the uploaded RGB source video is never sent to the API.

Enable **Qwen API 备用教师** in **系统设置**, then paste the API Key into its
write-only password field and save.  It is stored separately on the machine
running the Web service with owner-only permissions; subsequent reads and API
responses never return it.  Leaving the field empty preserves an existing key;
the page also provides an explicit clear option.  For server deployments, an
environment key remains preferred over the saved Web key.

```bash
export DASHSCOPE_API_KEY='your-key-here'
bash scripts/run_local_visualization.sh --web-python /path/to/web-env/bin/python \
  --rtmpose-python /path/to/pose-env/bin/python \
  --student-python /path/to/gcn-env/bin/python --device auto
```

The status pill reports whether a credential is available and whether it came
from the environment or protected Web settings, never the credential itself.
Teacher artifacts record `execution: qwen_api_fallback`, the selected model and
a response hash for audit, but never the API key or raw provider response.

## Pose extraction backend

The Campus6 production path uses the downloaded `RTMDet-nano-person + RTMPose-S`
TensorRT FP16 engines. The two models keep the same normalized two-person
COCO-17 contract consumed by the Campus6 ProtoGCN, so historical skeleton
artifacts do not need to be overwritten. FP32 MMPose remains available for
diagnosis; the experimental INT8 backend is selected only when explicitly
configured.

The validated FP16 package is kept at the paths below (or at
`DAHUA_POSE_FP16_MODEL_ROOT`):

```text
models/pose/fp16/
├── rtmdet_nano_person/deploy.json
├── rtmdet_nano_person/end2end.engine
├── rtmpose_s/deploy.json
└── rtmpose_s/end2end.engine
```

Install the tested Python environment with
`python -m pip install -r requirements/pose_fp16.txt` after installing the
TensorRT distribution compatible with the target CUDA/Python ABI. Source
`configs/pose/rtmpose_fp16.env.example` (or export the same variables) so the
launcher selects `DAHUA_CAMPUS6_BACKEND=rtmpose17`,
`DAHUA_POSE_BACKEND=tensorrt_fp16`, and `DAHUA_POSE_DEVICE=cuda:0`.
Switching this setting affects only later video-to-skeleton jobs; existing
COCO-17 skeleton artifacts remain unchanged.

## Run

The production Web student loads the accepted M1KD QAT INT8 checkpoint through
the portable quantized-state loader. The current PyTorch path restores those
weights into the CUDA inference graph; native INT8 kernel acceleration requires
a TensorRT, RKNN, or equivalent backend and is not claimed here.

```bash
export DAHUA_CODE_ROOT="$PWD"
export DAHUA_DATA_ROOT="$PWD/runtime"
export DAHUA_CAMPUS6_DEPLOY_CONFIG="$PWD/third_party/ProtoGCN/configs/campus6/rtm_s_coco17_k400_2d_gap_full.py"
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
| `GET/PUT /api/qwen-remote` | 配置优先使用的服务器 Qwen SSH 通道。 |
| `GET/PUT /api/qwen-api` | 配置非密钥的 Qwen OpenAI 兼容 API 备用通道；密钥仅由环境变量读取。 |
| `GET /api/samples`、`/api/samples/{sample_id}`、`/api/samples/{sample_id}/prediction` | 样本、预测和审核详情。 |
| `GET /api/samples/{sample_id}/media/pose` | 匿名骨架视频。 |
| `GET /api/hard-samples` | 待审核难例队列。 |
| `POST /api/samples/{sample_id}/jobs`、`GET /api/jobs/{job_id}` | 提交或查询处理任务。 |
| `POST /api/gcn/upload` | 上传本机视频并启动本机 RTMDet/RTMPose → ProtoGCN；仅门控触发时调用远端 Qwen。 |
| `POST /api/samples/{sample_id}/review`、`POST /api/samples/{sample_id}/undo`、`POST /api/reviews/undo-last` | 提交或撤销审核。 |
| `POST /api/samples/import-manifest`、`POST /api/datasets/export` | 导入样本或导出审核数据。 |

未知的非 `/api` 路径由 `visualization/` 静态页面处理；前端不接收原始 RGB 视频，
只请求服务器派生的骨架媒体。

### 本机 GCN 推理与 50 条增量闭环

“GCN 推理可视化”页面把 RGB 视频上传到**运行 Web 的本机**。本机依次执行
RTMDet/RTMPose COCO-17 和 Campus6 ProtoGCN；Qwen 不是本机依赖，只有门控触发时
才通过已保存的 SSH 配置向服务器传输骨架特征、骨架视频和学生结果。

人工审核确认的难例（显式 `human_reviewed_hard_sample`，或人工标签与学生标签不一致）
会复制到 `dataset/campus_increment/`。该目录始终保有与基线相同的
`annotations_with_all.pkl` 形态；后台每分钟检查一次，达到或超过 50 条时将**全部当前
待处理难例**冻结，使用固定的分组、分层 70/15/15 划分生成训练输入。若一分钟检查时
已有 53 条，则 53 条会一起进入同一轮，不会只上传前 50 条。

启用远程训练时，冻结的 `training_annotations_with_all.pkl` 上传到服务器；训练完成后先在
旧数据、新数据和合并快照上运行候选模型评估，再执行总体 Macro-F1、新样本 Macro-F1、旧数据
遗忘、危险类别召回、ECE 和部署大小 release gate。只有 gate 全部通过，候选权重才会注册并自动
切换 Production，随后本地才把该批次合并进 `dataset/campus_all/` 并清理已完成样本。任一步骤
失败或 gate 拒绝，`campus_increment` 和 round 工作目录都会保留以便追溯/重试。

默认评估命令由 `DAHUA_STUDENT_PYTHON` 调用 `dahua_cup.pipeline.evaluate_incremental`；也可用
`DAHUA_INCREMENTAL_EVALUATE_COMMAND` 覆盖（支持 `{config}`、`{candidate_checkpoint}`、
`{baseline_checkpoint}`、`{evaluation_annotation}`、`{old_annotation}`、`{new_annotation}`、
`{metrics_file}`、`{work_dir}` 占位符）。gate 阈值可通过 `DAHUA_INCREMENTAL_MIN_GLOBAL_F1_DELTA`、
`DAHUA_INCREMENTAL_MIN_NEW_F1_DELTA`、`DAHUA_INCREMENTAL_MAX_OLD_F1_DROP`、
`DAHUA_INCREMENTAL_MIN_DANGEROUS_RECALL_DELTA`、`DAHUA_INCREMENTAL_MAX_ECE_INCREASE` 和
`DAHUA_INCREMENTAL_MAX_EDGE_SIZE_BYTES` 调整。可用 `DAHUA_DATASET_ROOT`、
`DAHUA_INCREMENTAL_BATCH_SIZE` 和 `DAHUA_INCREMENTAL_TRAIN_COMMAND` 覆盖默认位置、阈值和训练命令。

如果已在 **系统设置** 启用 Qwen SSH 连接，达到 50 条时会复用同一受信服务器：上传冻结的
`training_annotations_with_all.pkl`，在服务器端自动挑选一张空闲 GPU（至少 20 GiB 空闲、利用率
不高于 10%）启动 10 epoch ProtoGCN 微调。远程训练是上传确认后的后台步骤；本地视频推理保持
可用。可用
`DAHUA_INCREMENTAL_REMOTE_ENABLED=0` 强制使用本地训练；远程 Python、初始 FP32 权重和 GPU 可通过
`DAHUA_INCREMENTAL_REMOTE_PYTHON`、`DAHUA_INCREMENTAL_REMOTE_INIT_CHECKPOINT`、
`DAHUA_INCREMENTAL_REMOTE_GPU_ID` 覆盖。
