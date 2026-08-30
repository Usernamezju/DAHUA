# 大华杯“校园行为语义理解”算法框架完整实施方案

> 文档状态：V1.2；2026-07-19 起姿态链路改为 MediaPipe，2026-07-28 完成源码目录归并
> 适用代码根目录：`/workspace/code/DAHUA`  
> 适用数据根目录：`/workspace/data/xzz_data/DAHUA`  
> 下载模型权重根目录：`/workspace/data/xzz_data/DAHUA/models`  
> 训练模型输出根目录：`/workspace/data/xzz_data/DAHUA/experiments/pipeline`  
> 当前选定骨架行为模型：**ProtoGCN**（BlockGCN 保留为后续对照实验）
>
> 本文是实施方案及历史设计记录，其中“缺口”描述对应起草时状态，
> 不作为当前代码完成度审计；当前可运行入口与已验证能力以
> `README.md`、现有代码和测试结果为准。
>
> 2026-08-03 运行配置已从 Qwen3-VL-32B BF16 调整为
> **Qwen3-VL-8B-Instruct FP16**。下文关于 32B 的容量估算作为历史方案保留，
> 当前启动与验收参数以 `docs/user_manual.md` 为准。

## 1. 文档目标与关键结论

本文只定义实施方案，不直接修改现有算法代码。目标是把比赛要求拆成可以开发、测试、部署和演示的工程任务，并尽量复用当前目录中的代码及官方开源实现。

建议最终采用“边缘端小模型 + 服务器端大模型教师 + 数据闭环”的双路径架构：

```text
边缘推理路径（实时、隐私优先）
视频
  -> MediaPipe Pose Landmarker（最多 2 人）
  -> MediaPipe33 世界坐标映射为 NTU25 3D 骨架
  -> 置信度过滤与双人槽位关联
  -> ProtoGCN NTU120 Bone 学生模型
  -> 类别、置信度、结构化证据

服务器教师路径（离线或低频触发）
结构化图摘要 + 姿态渲染视频 + 可选的人脸模糊视频片段
  -> Qwen3-VL-32B-Instruct
  -> 类别分布、语义理由、反证、质量判断
  -> 伪标签融合筛选与人工复核
  -> 增量训练/蒸馏
  -> 新版 ProtoGCN 学生模型
```

关键技术决策如下：

1. **当前冒烟链路使用 MediaPipe 33 点世界坐标映射 NTU25，并加载官方 NTU120 XSub Bone 权重。** MediaPipe 与 Kinect 存在传感器域差异，因此该结果只验证端到端工程链路；校园六分类仍需使用同一 MediaPipe 特征分布微调。
3. **大模型只作为服务器端教师，不进入边缘端 50 MB 限制。** 该口径需要向赛事方书面确认。即使如此，仍需实测检测、姿态、GCN 三个边缘模型的总文件大小；不能只计算 ProtoGCN。
4. **伪标签不能仅依据 Qwen 自报的 confidence。** 必须融合多次推理一致性、学生/教师一致性、姿态质量、时序稳定性，并把阈值在验证集上标定。
5. **隐私默认采用 pose-only。** 原视频只在提取阶段短暂存在；送入大模型的常规输入为骨架渲染视频和图摘要。确需使用 RGB 时，先进行人脸模糊并配置自动清理期限。

## 2. 当前代码资产与差距

### 2.1 可直接保留的目录

| 目录 | 当前作用 | 处理原则 |
|---|---|---|
| `gcn_models/ProtoGCN/` | 选定的 GCN 算法仓库 | 保留上游结构，只新增配置、数据转换脚本和适配器 |
| `gcn_models/BlockGCN/` | 候选对照模型 | 暂不纳入主流水线，后期仅做准确率/大小/速度对照 |
| `gcn_models/CTR-GCN/`、`gcn_models/2s-AGCN/`、`gcn_models/infogcn/`、`gcn_models/GAP/` | 已有骨架模型及实验 | 保留为基线，避免在主方案阶段改动 |
| `dahua_cup/feature_extraction/` | 已有四个视频特征模块 | 继续复用，后续补统一数据聚合器和命令入口 |
| `dahua_cup/semantic_teacher/` | Qwen、Prompt、伪标签骨架 | 在现有子目录内补全，不另起重复工程 |
| `dahua_cup/scripts/` | 分析和辅助脚本 | 保留；新增脚本需按职责放置 |

### 2.2 已有特征提取模块

| 文件 | 输入 | 输出 | 现状 |
|---|---|---|---|
| `dahua_cup/feature_extraction/mediapipe_pose.py` | MediaPipe33 姿态 | NTU25 `(x,y,z,score,mask)` | 已实现关节映射、虚拟关节和双人槽位关联 |
| `dahua_cup/feature_extraction/keypoint_filter.py` | 带 ID 的关键点序列 | 平滑关键点 | 已有 One Euro Filter，需补短缺口插值和长缺口屏蔽 |

### 2.3 `semantic_teacher` 起草时缺口（历史记录）

现有 `behavior_prompt.py` 已定义六类行为：正常行走、正常奔跑、嬉戏追逐、嬉戏推搡、冲突追逐、冲突推搡；`qwen_api.py` 已能调用本地 Qwen。以下部分仍需工程化：

- `dahua_cup/semantic_teacher/configs/config.yaml`、`distillation/distill.py`、`incremental/incremental_train.py`、`inference/infer.py` 和多个工具文件为空。
- 视频目录、输出目录、模型 snapshot 和 GPU 编号存在硬编码。
- 现有伪标签筛选只读取大模型的单一 `confidence`，没有姿态质量、一致性或版本溯源。
- Prompt 只要求单标签，没有类别概率分布、反证、证据字段和 JSON Schema 校验。
- `pseudo_labels/` 与 `dataset/accepted|review/` 存在两套输出位置和格式，需要统一。
- 尚无图构建、ProtoGCN 适配、统一推理入口、模型注册表、回滚机制和系统测试。

## 3. 总体系统架构

### 3.1 在线边缘推理

1. 解码视频，按照目标 FPS 采样。
2. MediaPipe Pose Landmarker 同时检测、估计并跟踪最多两个人的 33 点姿态。
3. 使用归一化髋中心最小位移把检测结果关联到稳定的双人槽位。
4. 使用世界坐标生成 NTU25 3D 骨架，并传播 visibility 和有效 mask。
5. 对低可信点置零；后续校园微调链路可增加短缺口插值和平滑。
6. 聚合每个人的中心轨迹、速度、加速度、方向变化、两人距离和接触持续时间。
7. 构造 ProtoGCN 所需的张量和可解释语义图。
8. ProtoGCN 输出六类 logits、softmax 概率和中间嵌入。
9. 证据引擎把数值特征映射成可核查的短句，例如“双方距离在 1.2 秒内持续缩短，随后连续近距离接触 0.8 秒”。
10. 输出 JSON；可选渲染带骨架、ID、标签和时间线的演示视频。

### 3.2 离线教师与数据闭环

1. 对无标签校园视频运行同一特征流水线。
2. 选择低置信、高熵、类别稀缺、遮挡或学生/规则冲突样本。
3. 将结构化摘要、姿态渲染视频和可选模糊 RGB 片段提交 Qwen。
4. 对不确定样本更换时间采样或 Prompt 运行两次，计算一致性。
5. 融合大模型分布、学生分布、姿态质量和时序一致性。
6. 自动接受高质量伪标签，中间样本进入人工复核，低质量样本拒绝。
7. 生成不可变的数据集版本和训练 manifest。
8. 采用真实标签、伪标签、历史回放样本和难例进行增量训练。
9. 新模型通过准确率、遗忘率、鲁棒性、速度和大小门槛后进入灰度发布，否则回滚。

### 3.3 进程与环境边界

MediaPipe 与 ProtoGCN/PYSKL 所依赖的旧版 MMCV 可以放在不同 Python 环境，通过 NPZ 契约解耦：

| 环境 | 建议路径 | 核心依赖 | 运行内容 |
|---|---|---|---|
| 姿态环境 | `/workspace/code/envs/pose_env` | MediaPipe、OpenCV、NumPy | 姿态和 NTU25 特征导出 |
| GCN 环境 | `/workspace/code/envs/protogcn_env` | ProtoGCN 仓库锁定的 PyTorch/MMCV 版本 | GCN 训练、评估和导出 |
| 大模型环境 | `/workspace/code/envs/llm_env` | Transformers、Qwen-VL 工具、Accelerate | 教师推理和伪标签生成 |
| 演示环境 | `/workspace/code/envs/demo_env`（可合并） | Gradio/OpenCV/ONNX Runtime | Web 展示和部署模型推理 |

模块之间通过版本化的 `.npz`、`.jsonl` 和 manifest 文件交换数据。这样可以避免依赖冲突，也便于失败重跑和结果追溯。

## 4. 建议目录组织

第三方 GCN 仓库与比赛自研工程分目录管理；不修改第三方仓库核心代码：

```text
/workspace/code/DAHUA/
├── gcn_models/
│   ├── ProtoGCN/                     # 上游算法仓库，主 GCN
│   ├── BlockGCN/                     # 后备对照，不接主流水线
│   ├── CTR-GCN/ 2s-AGCN/ infogcn/    # 现有对照基线
│   └── GAP/                          # GAP 基线及 NTU120 标签表
├── dahua_cup/                        # 本届比赛自研工程
│   ├── backend/                       # Web API、任务调度和审核存储
│   ├── feature_extraction/            # 姿态、轨迹、图和渲染
│   ├── semantic_teacher/              # Qwen、Prompt、伪标签和增量学习
│   ├── pipeline/                      # 只负责编排
│   ├── configs/campus/                # 统一配置入口
│   ├── visualization/                 # Web 展示与复核
│   ├── tests/                         # 自研部分测试，不侵入第三方仓库
│   ├── scripts/                       # 环境、下载校验、导出和 benchmark
│   └── requirements-web.txt
├── pytest.ini                         # 仓库级测试入口，仅收集 dahua_cup/tests
└── README.md                          # 统一运行说明
```

新增代码只能通过公开接口调用 `gcn_models/ProtoGCN/` 和其他第三方目录。若必须修改上游文件，应记录补丁文件和上游 commit，避免后续无法升级或复现实验。

## 5. 统一数据与模型路径

### 5.1 服务器目录

```text
/workspace/data/xzz_data/DAHUA/
├── models/                          # 只保存下载的第三方模型权重和配套配置
│   ├── pose_models/
│   │   └── mediapipe/
│   ├── Qwen3-VL-32B-Instruct/
│   └── protogcn_pretrained/
├── datasets/
│   ├── raw/                    # 原始比赛/采集视频，只读
│   ├── processed/features/     # 每段视频的关键点、轨迹和质量指标
│   ├── processed/graphs/       # GCN 张量和语义图
│   ├── ntu120_2d/              # 官方 NTU120 2D 数据
│   ├── pseudo/                 # 按版本保存伪标签
│   ├── review/                 # 人工复核数据库和导出文件
│   └── replay/                 # 增量训练回放集索引
├── experiments/
│   └── pipeline/                    # 本项目所有训练输出模型的统一根目录
│       ├── protogcn_ntu120_2d/
│       │   ├── checkpoints/
│       │   ├── logs/
│       │   └── reports/
│       ├── protogcn_campus6/
│       │   ├── checkpoints/
│       │   ├── logs/
│       │   └── reports/
│       ├── distillation/
│       ├── incremental/
│       ├── deployment/
│       │   ├── onnx/
│       │   ├── tensorrt/
│       │   ├── edge_bundle/
│       │   └── reports/
│       └── registry/                # 训练模型版本、指标和发布状态
└── registry/
    ├── datasets/                    # 数据集版本，不存模型权重
    └── prompts/
```

### 5.2 配置原则

- 路径只允许出现在 `dahua_cup/configs/campus/paths.yaml` 或环境变量中。
- 推荐环境变量：`DAHUA_CODE_ROOT`、`DAHUA_DATA_ROOT`、`DAHUA_MODEL_ROOT`。
- `DAHUA_MODEL_ROOT` 固定指向下载权重目录 `/workspace/data/xzz_data/DAHUA/models`，不得用于保存本项目训练 checkpoint。
- 所有训练、蒸馏、增量学习和部署导出产物必须写入 `/workspace/data/xzz_data/DAHUA/experiments/pipeline/<任务子目录>`。
- GPU 通过 `CUDA_VISIBLE_DEVICES` 或配置传入
- 每次运行保存完整配置副本、git commit、依赖版本、输入 manifest 和随机种子。
- 大文件不提交 Git；只提交下载清单、SHA256 和许可证说明。

## 6. 模块一：时空特征提取（必选）

### 6.1 姿态检测：MediaPipe Pose Landmarker

**职责**：从每帧直接输出最多两个人的 33 点归一化坐标、世界坐标和 visibility。

当前默认使用 `pose_landmarker_heavy.task`。检测、presence、tracking 三个阈值分别配置；最终阈值应由校园验证集确定。

输出：

```json
{
  "frame_index": 42,
  "timestamp_ms": 1400,
  "detections": [
    {"slot": 0, "landmarks": 33, "mean_visibility": 0.91}
  ]
}
```

工程要求：

- 使用 VIDEO 模式和严格递增的毫秒时间戳。
- 保存 FPS 和原始帧尺寸。
- 单帧空姿态写入全零 mask，不让异常终止视频。
- 小目标和严重遮挡场景单独统计 pose coverage。

### 6.2 双人槽位关联

**职责**：跨帧保持人员 ID，避免两人接近或交叉时身份交换。

MediaPipe 内部负责姿态时序跟踪；导出层仍需把每帧返回顺序映射到固定槽位：

- 首帧按归一化髋中心从左到右排序。
- 后续帧在最多两个槽位间穷举，以髋中心位移最小为目标。
- 空帧输出全零骨架，下一次检出继续参考最近中心。
- 镜头切换时重置槽位状态。

### 6.3 骨架转换：MediaPipe33 到 NTU25

**职责**：把 MediaPipe 世界坐标的头、肩、肘、腕、手、髋、膝、踝和足部映射为 ProtoGCN 使用的 NTU RGB+D 25 关节布局。

每个关节同时保存 `x`、`y`、`z`、score 和 `valid_mask`。低于阈值的关节坐标置零，以符合 ProtoGCN 对缺失 3D 关节的约定。

### 6.4 One Euro Filter 与缺失点处理

按 `track_id + joint_id` 维护过滤器状态：

- 使用真实视频时间戳计算 `dt`，不能假定解码帧率完全稳定。
- 高速度动作自动减弱平滑，静止时增强平滑。
- 连续缺失不超过 3 至 5 帧时，可按前后可信点插值。
- 长缺失不填充虚假姿态，保持 mask，并降低样本质量分。
- track 消失或镜头切换后释放滤波器状态。

### 6.5 派生时空特征

对每个确认人员计算：

- bbox 中心、骨盆中心和归一化中心轨迹。
- 速度、加速度、运动方向、方向变化次数、折返频率。
- 身体尺度归一化后的关节速度和骨骼夹角变化。
- 躯干朝向、双腕相对躯干速度、手臂快速伸展等攻击动作代理量。
- 双腕靠近头胸区域、躯干后仰、双臂遮挡等防御姿态代理量。

对每对人员计算：

- 中心距离、身体尺度归一化距离和相对速度。
- 是否持续接近、远离、同向运动或一前一后运动。
- bbox/人体区域重叠、腕部到对方躯干距离、可能接触时长。
- 追逐持续时间、角色交换、环形/折返轨迹和共同运动方向。

这些量是“可解释性”的事实层，只描述可测现象，不直接声称“攻击意图”。意图由分类器和教师综合判断。

### 6.6 特征文件契约

每段 clip 生成一个压缩 `.npz` 和一个元数据 `.json`：

```text
keypoint       [M, T, 25, 3] float32
keypoint_score [M, T, 25]    float32
valid_mask     [M, T, 25]    bool
fps            scalar        float32
width          scalar        int32
height         scalar        int32
total_frames   scalar        int32
```

元数据至少包含：

```json
{
  "schema_version": "mediapipe_ntu25.v1",
  "sample_id": "sha256-or-stable-id",
  "source_hash": "...",
  "fps": 30.0,
  "width": 1920,
  "height": 1080,
  "start_ms": 0,
  "end_ms": 4000,
  "extractor_versions": {},
  "quality": {
    "pose_coverage": 0.93,
    "mean_joint_score": 0.84,
    "track_fragmentation": 0.05,
    "id_switch_estimate": 0
  }
}
```

## 7. 模块二：时空图构建（必选）

本方案同时维护两个图表示：供 GCN 计算的规则张量，以及供语义映射和解释使用的结构化图。两者共享同一 `sample_id` 和时间窗口。

### 7.1 ProtoGCN 计算图

MediaPipe33 通过明确的关节映射生成 NTU25，虚拟的脊柱、颈部和手部节点由相关 MediaPipe 点取均值。例如：

- NTU 节点 0（spine base）：MediaPipe 左右髋 23、24 的均值。
- NTU 节点 1（spine mid）：左右髋与左右肩 23、24、11、12 的均值。
- NTU 节点 2/20（neck/spine shoulder）：左右肩 11、12 的均值。
- NTU 左右手节点：腕、拇指、食指和小指相关点的均值。

节点顺序和 `gcn_models/ProtoGCN/protogcn/utils/graph.py` 的 `nturgb+d` 邻接矩阵保持一致。虚拟节点 score 取依赖点均值；低于阈值或坐标非有限时同步把 mask 置为无效。

GCN 输入统一为：

```text
[N, M, T, V, C]
N: batch
M: 最多人数，建议 2；可扩展到 4，但需评估显存和校园群体场景
T: 时间长度，例如 100 帧；通过均匀采样或滑动窗口固定
V: 25 个节点（nturgb+d）
C: x, y, z 世界坐标
```

当前冒烟链路直接使用 MediaPipe 世界坐标，并由官方配置中的 `PreNormalize3D` 进行中心归一化。校园微调必须继续使用同一坐标约定。

### 7.2 图节点与边

低层骨架图节点为关节，包含坐标、置信度、速度和 mask。边包括：

1. **空间骨骼边**：按 COCO 人体拓扑连接相邻关节。
2. **时间边**：同一 `track_id`、同一关节在相邻帧连接。
3. **对称边（可选）**：左右肩、肘、腕、髋、膝、踝之间的辅助连接。
4. **跨人交互边**：当两人距离低于身体尺度阈值时，连接腕到对方躯干、中心到中心等关系。
5. **自适应边**：由 ProtoGCN 学习，不替代明确的物理边。

### 7.3 语义时空图

语义图不把每一帧全部展开给大模型，而是按片段聚合：

```json
{
  "schema_version": "semantic_graph.v1",
  "sample_id": "...",
  "segments": [
    {"id": "s0", "start_ms": 0, "end_ms": 1200, "motion": "approaching"},
    {"id": "s1", "start_ms": 1200, "end_ms": 2300, "motion": "close_contact"}
  ],
  "persons": [
    {"id": 3, "mean_speed": 0.38, "turn_count": 2, "pose_coverage": 0.94},
    {"id": 8, "mean_speed": 0.34, "turn_count": 1, "pose_coverage": 0.89}
  ],
  "relations": [
    {"src": 3, "dst": 8, "type": "approach", "duration_ms": 900, "confidence": 0.86},
    {"src": 3, "dst": 8, "type": "possible_contact", "duration_ms": 620, "confidence": 0.71}
  ],
  "quality": {}
}
```

图关系必须由可复现的数值规则生成，并保存原始数值。文本只是展示层，不能是唯一证据。

### 7.4 窗口和多人策略

- 主窗口建议 3 至 5 秒，步长 1 至 2 秒；具体值用验证集选择。
- 一段视频可以产生多个 clip，视频级结果由时间加权投票或时序后处理合并。
- 优先选择交互最强的两人作为 `M=2`；其余人员作为环境统计，如“周围人数”和“围观持续时间”。
- 群体冲突需要另设 `M=4` 实验，不应在 V1 阶段盲目扩大输入。

## 8. ProtoGCN 数据与训练路线

### 8.1 为什么不用 17 点直接套 25 点权重

现有 NTU120 模型的 25 点来自 Kinect 3D 骨架，关节定义、维度和图邻接均与 COCO17 不同。把缺失关节补零或简单复制会造成：

- 训练和部署输入分布不同。
- 邻接关系的语义错误。
- 由单目 2D 估计伪 3D 会引入深度噪声。
- 预训练权重看似可加载，但结果不可解释且难以复现。

### 8.2 推荐主路线：NTU120-2D 预训练

1. 下载官方 MMAction2 的 `ntu120_2d.pkl`。
2. 将其适配为 ProtoGCN/PYSKL annotation 格式。
3. 使用 `layout='coco'` 或仓库已有的 `coco_new` 20 节点拓扑。
4. `num_class=120` 进行 NTU120-2D 预训练。


## 9. 模块三：语义映射层（必选）

### 9.1 输入内容

Prompt 不直接倾倒完整图 JSON，而是由模板生成固定结构：

1. 任务、闭集标签和标签定义。
2. 时间段摘要。
3. 人员轨迹、速度、折返、接触和姿态证据。
4. 数据质量和不可见信息。
5. ProtoGCN top-k 分布，而非只给 top-1。
6. 两到四个经过审核的 few-shot 示例。
7. 严格输出 Schema。

必须明确告诉大模型：低质量/不可见信息应写为 unknown，不得补造攻击、防御或接触事实。

### 9.2 输出 Schema

```json
{
  "schema_version": "teacher_output.v1",
  "sample_id": "...",
  "label": "conflict_push",
  "distribution": [
    {"label": "conflict_push", "probability": 0.62},
    {"label": "playful_push", "probability": 0.25}
  ],
  "confidence": 0.62,
  "evidence": [
    {"type": "sustained_contact", "segment_id": "s1", "description": "..."}
  ],
  "counter_evidence": ["未检测到明确防御姿态"],
  "reason": "...",
  "needs_review": true
}
```

要求：

- `distribution` 覆盖全部六类并归一化，或者至少返回 top-k 后由程序明确补齐。
- `evidence` 必须能引用 segment 或结构化特征。
- 使用 JSON Schema/Pydantic 校验；解析失败只允许有限次数重试。
- 保存 `prompt_version`、模型版本、采样帧索引、生成参数和原始响应 hash。
- 教师理由不能直接当作事实；界面展示时区分“测量证据”和“模型解释”。

### 9.3 Prompt 版本管理

每次 Prompt 改动创建版本，例如 `campus6-v1.2`，并在固定验证集上回归：

- JSON 合法率。
- 分类准确率和 macro-F1。
- 同一样本重复推理一致率。
- 证据引用正确率。
- 幻觉率：理由包含输入未提供事实的比例。

未通过回归的 Prompt 不进入伪标签生产流程。

## 10. 模块四：大模型推理（必选）

### 10.1 模型选择

主模型统一采用官方 `Qwen/Qwen3-VL-32B-Instruct`，本地目录为 `/workspace/data/xzz_data/DAHUA/models/Qwen3-VL-32B-Instruct/`。它只作为服务器端离线教师，不部署到边缘设备，也不计入边缘端 50 MB 模型包。

选择 32B 后，原有 `dahua_cup/semantic_teacher/api/qwen_api.py` 不能只修改路径：需要把模型类、Processor、视频消息格式和依赖切换到 Qwen3-VL 官方接口。推荐使用 `Qwen3VLForConditionalGeneration`、`AutoProcessor` 和官方 Qwen-VL 工具；具体调用代码以下载时固定的官方模型 revision 为准。

资源规划：

- 32B 参数仅按 BF16 权重理论计算约需 64 GB，实际加载还包括视觉编码器、KV cache、CUDA kernel 和输入视频张量，必须预留额外显存。
- BF16/FP16 基线优先使用 `device_map="auto"` 跨多张 GPU 分配。短视频输入可先在 1 张 80 GB 或至少 2 张 48 GB 级别 GPU 上进行试运行，但这只是资源规划起点，最终以服务器实测为准。
- 官方建议启用 FlashAttention 2，以改善多图和视频场景的速度与显存占用；环境安装前需确认 GPU、CUDA 和 PyTorch 兼容。
- 若服务器无法稳定加载 BF16，可评估官方或经过验证的 FP8/4-bit 版本，但量化模型必须单独做六类准确率、概率校准和 JSON 合法率回归，不能默认与原模型等价。
- 模型下载和转换需要较大临时空间；建议模型目录至少预留 100 GB，并避免在容器可写层重复保存 Hugging Face cache。

建议输入优先级：

1. **默认**：姿态渲染视频 + 结构化图摘要。
2. **不确定样本**：姿态渲染视频 + 人脸模糊 RGB 稀疏帧 + 图摘要。
3. **纯结构化回退**：只有图摘要和学生输出，用于显存不足或隐私限制场景。

### 10.2 推理策略

- 生产伪标签时使用 Instruct 版本和确定性或低温度解码，不在主流程使用 Thinking 版本，避免输出结构和延迟不稳定。
- 普通高质量样本推理一次；高熵或冲突样本按两种时间采样运行两次。
- 若两次 top-1 不一致或 Jensen-Shannon divergence 超阈值，进入复核，不自动接受。
- 将长视频切成短片段，限制输入帧数，避免显存峰值和无关上下文。
- 使用 `local_files_only=True` 并固定 Hugging Face revision，运行阶段不得临时联网下载模型。
- 使用官方支持 Qwen3-VL 的 Transformers 版本；安装成功后立即冻结为环境 lock 文件，不让后续升级改变推理结果。
- 首次上线前分别测试纯结构化输入、姿态视频输入、模糊 RGB 输入的峰值显存，按结果限制最大帧数、分辨率和并发数。
- 32B 教师默认采用单任务队列或受控小并发，OOM 后自动降低输入帧数重试，不与 ProtoGCN 训练抢占同一组 GPU。
- 支持断点续跑；按 `sample_id + model_version + prompt_version` 做幂等缓存。
- 捕获 OOM、超时、无效 JSON 和视频解码错误，失败样本进入独立错误队列。

### 10.3 教师不是绝对真值

Qwen 生成结果属于弱监督教师信号。不得用以下方式构成循环自证：将学生 top-1 强提示给大模型，大模型复述学生结果，再把结果当独立教师。Prompt 应给出学生完整 top-k 并明确允许纠正；筛选时将大模型一致性和学生一致性分开计分。

## 11. 模块五：伪标签筛选与人工复核（必选）

### 11.1 质量分

建议初始综合分：

```text
S = 0.30 * llm_consistency
  + 0.25 * teacher_probability
  + 0.20 * student_teacher_agreement
  + 0.15 * pose_quality
  + 0.10 * temporal_stability
```

其中：

- `llm_consistency`：多次采样结果的一致性或分布相似度。
- `teacher_probability`：教师目标类别概率，需做验证集校准。
- `student_teacher_agreement`：两分布相似度，不只是 top-1 是否相同。
- `pose_quality`：关键点覆盖率、均值分数、轨迹完整度、ID 切换等。
- `temporal_stability`：相邻窗口预测是否稳定。

初始分流建议仅作为待标定起点：

| 条件 | 状态 |
|---|---|
| `S >= 0.85`、姿态覆盖率不低于 0.70、无关键冲突 | 自动接受 |
| `0.55 <= S < 0.85`，或教师/学生冲突，或类别稀缺 | 人工复核 |
| `S < 0.55`、Schema 无效、追踪/姿态严重失败 | 拒绝或重新提取 |

阈值必须在人工标注验证集上通过 precision-coverage 曲线确定。比赛演示应展示“接受伪标签的精确率”和“自动覆盖率”之间的关系，而不是只报接受数量。

### 11.2 规则冲突

下列情况强制进入复核：

- 教师 top-1 与学生 top-1 不同且双方均高置信。
- “推搡”类别但未检测到任何近距离/接触证据，且姿态质量足够好。
- “追逐”类别但两人连续相对运动时长不足。
- 两次教师推理标签不一致。
- ID switch、严重遮挡或有效关键点低于质量门槛。
- 罕见类别样本，即使高分也按一定比例抽检。

这些规则不直接改标签，只决定自动接受还是复核。

### 11.3 人工复核队列

推荐使用 SQLite 保存队列和审计信息，不用单个 JSON 文件承载多人修改：

```text
samples(sample_id, artifact_path, suggested_label, score, status, priority, versions...)
reviews(review_id, sample_id, reviewer, final_label, reason_code, note, created_at)
events(event_id, sample_id, old_status, new_status, actor, created_at)
```

复核界面提供：

- 人脸模糊视频和骨架叠加。
- 六类单选、无法判断、数据损坏。
- 教师/学生分布，但默认先让复核者独立判断，减少锚定偏差。
- 测量证据时间线和可跳转片段。
- 冲突原因、姿态质量、数据/模型/Prompt 版本。

至少对 10% 自动接受样本做随机抽检；对低频类别提高抽检比例。

### 11.4 伪标签记录

每条记录必须包含：

```json
{
  "schema_version": "pseudo_label.v1",
  "sample_id": "...",
  "label": "playful_chase",
  "soft_label": [0.01, 0.03, 0.62, 0.08, 0.21, 0.05],
  "quality_score": 0.88,
  "status": "accepted",
  "source": "qwen_teacher",
  "feature_version": "...",
  "student_model_version": "...",
  "teacher_model_version": "...",
  "prompt_version": "...",
  "filter_version": "...",
  "review": null,
  "created_at": "..."
}
```

## 12. 模块六：知识蒸馏（必选）

### 12.1 推荐蒸馏目标

Qwen 和 GCN 的 token/关节空间不同，不适合直接做原始 attention 对齐。主路线采用类别分布的 logits 蒸馏：

```text
L_total = λh * CE(y_hard, p_student)
        + λp * w_quality * CE(y_pseudo, p_student)
        + λk * w_quality * T² * KL(q_teacher^T || p_student^T)
```

- `y_hard`：人工真实标签。
- `y_pseudo`：筛选后的伪标签。
- `q_teacher`：Qwen 输出的六类分布。
- `T`：蒸馏温度。
- `w_quality`：由伪标签质量分映射的样本权重。

### 12.2 训练批次组成

每个 batch 建议混合：

- 40% 人工真实标签。
- 30% 高质量伪标签。
- 20% 历史回放样本。
- 10% 难例或低频类别过采样。

比例应按数据量和消融结果调整，但人工标签不能被大量伪标签淹没。伪标签只用于训练，固定人工验证集不可被污染。

### 12.3 训练保护

- 教师权重冻结。
- 保存伪标签分布，而不只保存 argmax。
- 对教师概率进行温度校准。
- 每轮监控类别分布，防止模型坍缩到正常行走/奔跑。
- 早停依据 macro-F1 和危险类别召回，不只依据总准确率。
- 对“嬉戏/冲突”成对类别报告混淆矩阵。
- 每个实验保存损失权重、温度、随机种子和数据 manifest。

## 13. 模块七：增量学习与自进化（必选）

### 13.1 闭环批次流程

不采用“来一个样本就在线改一次权重”。推荐按版本批处理：

```text
采集新视频
 -> 特征/图提取
 -> 难例选择
 -> Qwen 伪标签
 -> 自动筛选 + 人工复核
 -> 生成 dataset_vNN
 -> 混合历史 replay 训练 candidate_vNN
 -> 离线验收
 -> 灰度/影子运行
 -> 发布或回滚
```

### 13.2 难例挖掘

优先送教师和复核的样本：

- 学生预测熵高或 top-1/top-2 间隔小。
- 教师和学生分布差异大。
- 嬉戏与冲突子类混淆。
- 追踪 ID 交换、遮挡、弱光、侧视或俯视。
- 相邻窗口标签频繁切换。
- 低频类别或新场景/新摄像头。
- 业务规则与模型输出冲突。

### 13.3 防灾难性遗忘

- 维护按类别、场景、摄像头和难度分层的 replay buffer。
- 新模型对旧模型做 logits 蒸馏，约束旧数据行为。
- 分别报告旧验证集、新数据集和总体指标。
- 设最大遗忘率，例如旧集 macro-F1 下降不得超过 1 至 2 个百分点；最终阈值根据赛事目标确定。
- 新模型未通过门槛时不得覆盖生产模型。

### 13.4 数据和模型注册表

每个数据版本保存：来源样本、标签来源、筛选版本、复核记录和 SHA256。每个模型版本保存：

```json
{
  "model_id": "protogcn-campus6-v0.4.0",
  "parent_model_id": "protogcn-campus6-v0.3.0",
  "dataset_id": "campus6-dataset-v12",
  "config_hash": "...",
  "checkpoint_hash": "...",
  "metrics": {},
  "edge_size_bytes": 0,
  "latency": {},
  "status": "candidate"
}
```

发布状态至少为 `candidate -> validated -> canary -> production -> archived`，并保存生产版软链接或显式 manifest，支持一分钟内回滚。


## 14. 可解释性设计

最终输出分三层：

1. **模型结果**：标签和六类概率。
2. **测量证据**：由轨迹/骨架确定的数值事实。
3. **语义解释**：模板或 Qwen 对事实的总结。

推荐输出：

```json
{
  "label": "playful_chase",
  "confidence": 0.81,
  "measured_evidence": [
    "两人连续同向快速移动 2.4 秒",
    "最近距离缩短后多次折返",
    "未检测到持续近距离接触"
  ],
  "semantic_reason": "判断为嬉戏追逐，因为双方存在持续追逐和折返，但缺少攻击性接触及明确防御姿态。",
  "limitations": ["第二人的腕部在 18% 帧中被遮挡"]
}
```

不允许生成输入中不存在的“拳打”“恐惧”“故意”等结论。所有解释模板都需能追溯到特征字段或教师 evidence 引用。

## 20. 模型权重和大文件清单

### 20.1 必需下载

| 资源 | 建议服务器位置 | 说明 |
|---|---|---|
| MediaPipe Pose Landmarker task | `/workspace/data/xzz_data/DAHUA/models/pose_models/mediapipe/` | 33 点姿态 |
| ProtoGCN NTU120 XSub Bone `b_1` | `/workspace/data/xzz_data/DAHUA/models/protogcn_pretrained/ntu120_xsub_bone_b1/` | 120 类冒烟和微调初始化 |
| `ntu120_2d.pkl` | `/workspace/data/xzz_data/DAHUA/datasets/ntu120_2d/` | ProtoGCN 2D 预训练/验证 |
| Qwen3-VL-32B-Instruct 完整 snapshot | `/workspace/data/xzz_data/DAHUA/models/Qwen3-VL-32B-Instruct/` | 服务器已有 63 GB 完整权重和 14 个分片；从公共目录复制/复用并校验，不再重复下载 |

已知官方权重地址：

```text
NTU120 2D annotation:
https://download.openmmlab.com/mmaction/v1.0/skeleton/data/ntu120_2d.pkl

Qwen3-VL-32B-Instruct:
https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct
```

下载前应以官方模型库页面核对 config/checkpoint 配对关系；文件名相近不代表结构相同。

在可联网机器上下载 Qwen 完整 snapshot 的示例：

```bash
hf download Qwen/Qwen3-VL-32B-Instruct \
  --local-dir ./Qwen3-VL-32B-Instruct
```

必须确认 14 个 safetensors 分片、索引、配置、Processor/Tokenizer 文件全部存在；不要只传输单个权重分片。下载时记录实际 revision，并在服务器配置中固定该 revision。

### 20.2 可选下载或训练产物

- ProtoGCN 官方 Kinetics-400 预训练权重：用于 COCO 图初始化对照。
- 六分类 ProtoGCN checkpoint：由本项目训练，放入 `/workspace/data/xzz_data/DAHUA/experiments/pipeline/protogcn_campus6/checkpoints/`，不放入下载模型目录。
- INT8 calibration 数据：从真实训练集按场景抽取，不含测试集。
- BlockGCN 权重：只有在 ProtoGCN 不满足大小/速度/精度目标时再准备。

### 20.3 无网络服务器的传输方式

服务器容器已表现为无法访问公网，因此应在可联网机器下载并校验，再通过宿主机别名传输。示例：

```bash
rsync -av --progress ./pose_models/ \
  Lab133_Docker:/workspace/data/xzz_data/DAHUA/models/pose_models/

rsync -av --progress ./ntu120_2d.pkl \
  Lab133_Docker:/workspace/data/xzz_data/DAHUA/datasets/ntu120_2d/

rsync -av --progress ./Qwen3-VL-32B-Instruct/ \
  Lab133_Docker:/workspace/data/xzz_data/DAHUA/models/Qwen3-VL-32B-Instruct/
```

传输后在两端执行 `sha256sum`，保存到 `dahua_cup/scripts/checksums/models.sha256`。同步代码时不要使用 `--delete`，避免误删服务器实验或用户改动。

## 21. 配置示例

`dahua_cup/configs/campus/paths.yaml`：

```yaml
code_root: /workspace/code/DAHUA
competition_root: ${code_root}/dahua_cup
gcn_root: ${code_root}/gcn_models
data_root: /workspace/data/xzz_data/DAHUA
model_root: /workspace/data/xzz_data/DAHUA/models
experiment_root: /workspace/data/xzz_data/DAHUA/experiments/pipeline

pose:
  backend: mediapipe
  model: ${model_root}/pose_models/mediapipe/pose_landmarker_heavy.task
  num_poses: 2
student:
  config: ${gcn_root}/ProtoGCN/configs/ntu120_xsub/b.py
  checkpoint: ${model_root}/protogcn_pretrained/ntu120_xsub_bone_b1/best_top1_acc_epoch_150.pth
  modality: bone
teacher:
  model_id: Qwen/Qwen3-VL-32B-Instruct
  model_dir: ${model_root}/Qwen3-VL-32B-Instruct
  revision: <downloaded-commit-id>
  dtype: bfloat16
  device_map: auto
  attn_implementation: flash_attention_2
  local_files_only: true
  max_concurrency: 1
outputs:
  features: ${data_root}/datasets/processed/features
  graphs: ${data_root}/datasets/processed/graphs
  protogcn_ntu120_2d: ${experiment_root}/protogcn_ntu120_2d
  protogcn_campus6: ${experiment_root}/protogcn_campus6
  distillation: ${experiment_root}/distillation
  incremental: ${experiment_root}/incremental
  deployment: ${experiment_root}/deployment
```

实际实现可使用 OmegaConf/Hydra 或项目已有配置机制解析变量；不要自行用字符串替换实现复杂配置。

## 22. API 和命令入口设计

最终对使用者暴露少量稳定命令：

```bash
# 1. 视频转特征/图
python -m dahua_cup.pipeline.extract_dataset \
  --config dahua_cup/configs/campus/models.yaml \
  --input-manifest /path/to/videos.jsonl

# 2. 生成并筛选伪标签
python -m dahua_cup.pipeline.generate_pseudo_dataset \
  --dataset-version campus-unlabeled-v1 \
  --prompt-version campus6-v1

# 3. 训练或蒸馏学生模型
python -m dahua_cup.pipeline.train_student \
  --config gcn_models/ProtoGCN/configs/campus6/j.py

# 4. 单视频端到端推理
python -m dahua_cup.pipeline.infer_video \
  --video /path/to/test.mp4 \
  --output /path/to/result.json

# 5. 闭环候选版本构建
python -m dahua_cup.pipeline.run_closed_loop \
  --base-model protogcn-campus6-v0.3.0 \
  --new-data campus-batch-v12
```

每个命令支持 `--dry-run`、明确退出码、结构化日志和断点续跑。pipeline 只编排各环境中的子命令，不复制底层算法逻辑。

## 24. 指标与最终验收标准

### 24.1 算法指标

- 六类 accuracy、macro-F1、per-class precision/recall/F1。
- 嬉戏/冲突两组混淆矩阵。
- 危险行为召回：冲突追逐、冲突推搡。
- Expected Calibration Error、Brier score 或可靠性图。
- 视频级和 clip 级指标分开报告。

### 24.2 提取和跟踪指标

- 人体检测 recall。
- 关键点覆盖率和平均置信度。
- IDF1/HOTA（有跟踪标注时）或 ID switch/轨迹断裂统计。
- 不同人数、距离和遮挡条件下的失败率。

### 24.3 伪标签指标

- 自动接受 precision。
- 自动覆盖率。
- 人工复核通过率和平均复核时间。
- 各类别接受数量和分布偏差。
- 教师重复推理一致率和 JSON 合法率。

### 24.4 工程指标

- 完整边缘 bundle `<= 50 MB`，以实际文件为准。
- 目标设备 FPS、P95 延迟、峰值内存/显存。
- 长时间运行无资源持续增长。
- 同一 manifest、配置和 seed 可复现实验。
- 任一结果可追溯到输入、特征、Prompt、数据集和模型版本。

## 28. 权威资料与实现依据

1. [MMAction2 Skeleton 数据准备：包含 NTU60/120 2D annotation](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/skeleton/README.md)
2. [MMAction2 NTU120 2D ST-GCN 官方配置](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/stgcn/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-2d.py)
3. [MMAction2 ST-GCN 模型结果与配置说明](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/stgcn/README.md)
4. [PYSKL：OpenMMLab Skeleton Action Recognition Toolbox](https://arxiv.org/abs/2205.09443)
5. [ProtoGCN：CVPR 2025 论文页面](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_Revealing_Key_Details_to_See_Differences_A_Novel_Prototypical_Perspective_CVPR_2025_paper.html)
6. [MediaPipe Pose Landmarker 文档](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)
7. [One Euro Filter 原论文](https://doi.org/10.1145/2207676.2208639)
10. [Qwen3-VL-32B-Instruct 官方模型页](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct)
11. [Qwen3-VL 官方 GitHub 仓库](https://github.com/QwenLM/Qwen3-VL)
12. [不同骨架拓扑 Joint Mapping 的研究](https://openaccess.thecvf.com/content/WACV2023/html/Kang_Efficient_Skeleton-Based_Action_Recognition_via_Joint-Mapping_Strategies_WACV_2023_paper.html)
