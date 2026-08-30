# 大华杯功能要求最终验收

本文把验收证据分为三个层级：

- **代码实现**：仓库存在可调用实现，并非只有设计文档或占位按钮。
- **自动验证**：单元/集成测试覆盖关键契约与分流逻辑。
- **真实验收**：使用真实视频、真实权重和独立人工测试集跑通并保存指标。

只有三个层级全部满足，才称为“完整验收通过”。代码仓库不分发数据集和
模型权重；服务器已恢复 562 个候选视频，但仍没有 Campus6 官方人工测试集，
因此不能把代码完成或 NTU120 联调等同于真实六分类效果已经验收。

## 一、当前结论

| 要求 | 代码实现 | 自动验证 | 真实验收 | 主要证据或缺口 |
|---|---|---|---|---|
| 时空特征提取 | ✅ | ✅ | ⚠️ | MediaPipe、NTU25、轨迹/速度/加速度、质量分数均已实现；需要恢复视频做真机回归 |
| 时空图构建 | ✅ | ✅ | ⚠️ | `semantic_graph.v1` 有人物节点、时空片段及人物关系；Qwen 工作者使用针对 NTU25 的轻量图摘要 |
| 语义映射层 | ✅ | ✅ | ⚠️ | Prompt 包含图、学生分布、标签约束、证据段和反证要求；需用真实 Qwen 检查输出稳定性 |
| 大模型伪标签 | ✅ | ✅（协议/Mock） | ⚠️ | Qwen3-VL-32B 已成功保存 1 条真机结果；当前选定 8B FP16 尚未做固定集回归 |
| 置信度筛选与人工队列 | ✅ | ✅ | ⚠️ | 多信号门控、离线挖掘、自动稀有类、SQLite 队列、Web 审核和回写已接通；阈值仍需人工集校准 |
| 知识蒸馏 | ✅ | ✅（损失/数据/触发） | ❌ | NTU120 Teacher-Student logits KL、伪标签质量加权和 CSC 已接入 ProtoGCN，并由 Web 后台自动触发；当前环境未实际训练 |
| 增量学习 | ✅ | ✅（数据/门禁） | ❌ | 服务启动后自动执行伪标签注入、难例、回放、微调条件判断、注册和安全发布；尚无一次真实闭环训练记录 |
| 可视化（可选） | ✅ | ✅ | ✅（工程链路） | Web 已加载 562 个候选视频，并生成视频、骨架、学生和教师产物 |
| 可解释性 | ✅ | ✅ | ⚠️ | 输出证据片段、关系、反证和简短理由；需人工抽检理由忠实性 |
| 可进化性 | ✅ | ✅ | ❌ | Web 后台与闭环中心已经自动调度完整链路，但还没有“基线→增量模型→指标提升”的真实报告 |
| 轻量化 ≤50MB | ⚠️ | ✅（完整包门禁） | ❌ | ProtoGCN 单文件 32.10 MiB；与 MediaPipe Heavy 合计约 61.35 MiB，需改用 Lite 后复测 |
| 鲁棒性 | ⚠️ | ✅（部分） | ❌ | 有旋转/翻转增强、平滑、插值、质量门控和 pose-only 输入；尚无遮挡/视角/光照分组指标 |
| 隐私保护 | ✅（pose-only） | ✅ | ⚠️ | 教师主链路使用骨架视频且不需上传原视频；`blur_faces_before_teacher` 目前只是配置，RGB 回退的人脸模糊未实现 |
| 代码规范与 README | ✅ | ✅ | ✅ | 模块化目录、配置、注释、README、闭环文档和回归测试均存在 |

所以准确表述是：**必选模块的工程代码基本实现，但整个项目还没有全部
完成真实验收。** 当前硬阻塞是数据缺失；恢复数据后还必须完成 Qwen
真机推理、六分类蒸馏/增量训练、模型压缩或导出、独立测试集评估。

## 二、一键代码验收

在仓库根目录执行：

```bash
python -m pip install -r dahua_cup/requirements-test.txt
python -m dahua_cup.scripts.verify_requirements
```

它会运行全部 Pytest、Python 编译、Web Shell/JavaScript 语法检查，以及
所有关键 CLI 的 `--help` 启动检查。也可以输出机器可读报告：

```bash
python -m dahua_cup.scripts.verify_requirements \
  --output /tmp/dahua-requirements-code.json
```

这一步成功只代表代码级验收，不代表 GPU 模型已经跑通。

服务器启动 Web 后，再执行运行时检查：

```bash
python -m dahua_cup.scripts.verify_requirements \
  --runtime \
  --data-root /workspace/data/xzz_data/DAHUA \
  --base-url http://127.0.0.1:8000 \
  --nanodet-config /path/to/nanodet-m.yml \
  --nanodet-checkpoint /path/to/nanodet-m.pth \
  --nanodet-python /path/to/nanodet/python \
  --edge-model /path/to/nanodet-m.pth \
  --output /workspace/data/xzz_data/DAHUA/runtime/requirements-runtime.json
```

NanoDet的配置、权重和Python参数不提供时对应检查会明确标记为`SKIP`；
提供后会验证文件、运行环境导入和Web的`person_localization`能力状态。

候选视频没有随代码仓库分发；数据目录为空时 `candidate_videos` 应当失败。
`--edge-model` 应对边缘包中的每个模型文件重复传入，验收程序按总大小判断，
不能用指标 JSON 中的自报数字代替实际文件。

## 三、真实视频端到端验收

### 1. 恢复最小数据

准备至少三段授权视频：单人正常动作、多人交互、严重遮挡或骨架失败。
生成 manifest：

```bash
python dahua_cup/scripts/build_manual_label_manifest.py \
  --root /workspace/data/xzz_data/DAHUA/datasets/vedio \
  --output /workspace/data/xzz_data/DAHUA/datasets/vedio/manual_label_manifest.csv
```

启动服务。阈值默认从 YAML 读取，命令行可覆盖：

```bash
bash dahua_cup/scripts/run_visualization.sh \
  --routing-config dahua_cup/configs/campus/teacher_routing.yaml \
  --confidence-threshold 0.70 \
  --margin-threshold 0.15 \
  --pose-quality-threshold 0.70
```

### 2. 特征、图、学生和 Qwen

在 Web 对每类样本执行“生成预览→提取骨架→运行完整推理”，检查：

1. `artifacts/features/<sample>.npz` 含 `keypoint`、`keypoint_score`、
   `valid_mask`、`fps`；
2. `artifacts/predictions/<sample>.json` 含 NTU120 Top-5、
   `label_space_size=120`、姿态质量和 `teacher_gate`；
3. 高置信且大间隔、骨架良好时 `teacher_gate.route=use_student`；
4. 低置信或小间隔时 `teacher_gate.route=call_teacher`；
5. 可用但低质量骨架仍运行 ProtoGCN，并作为风险信号触发 Qwen/人工复核；
   只有完全无可用姿态时才停止 GCN 并要求重提取或人工处理；
6. Qwen 结果含闭集标签、分布、置信度、证据/反证片段和理由；
7. 学生与 Qwen 高置信冲突时队列优先级为最高级。

为了只测试路由而强制让合格骨架进入 Qwen，可以临时启动：

```bash
bash dahua_cup/scripts/run_visualization.sh \
  --confidence-threshold 1.0 \
  --margin-threshold 1.0
```

这只是验收设置，不能作为生产阈值。

### 3. NTU120 自动闭环

正常启动 Web 后，伪标签收集、离线难例挖掘和训练条件判断已经自动运行，
不再要求日常手动执行 `generate_pseudo_dataset` 或 `mine_candidates`：

```bash
bash dahua_cup/scripts/run_visualization.sh
curl -fsSL http://127.0.0.1:8000/api/evolution
```

只有满足数据量、基础标注、训练环境、当前 checkpoint 和冷却期门槛后才
开始 NTU120 蒸馏。训练候选没有通过同一 XSub 验证集的平均类别准确率、
Top-1 非回退和 50 MiB 三重门禁时，原权重保持不变。详细设计和产物见
[`ntu120_auto_evolution.md`](ntu120_auto_evolution.md)。

### 4. 手工重跑 NTU120 难例挖掘

无需再写大量参数，默认读取
`configs/campus/hard_mining.yaml`：

```bash
python -m dahua_cup.pipeline.mine_candidates \
  --prediction-dir /workspace/data/xzz_data/DAHUA/runtime/visualization/artifacts/predictions \
  --teacher-dir /workspace/data/xzz_data/DAHUA/runtime/visualization/artifacts/teachers \
  --output /workspace/data/xzz_data/DAHUA/runtime/hard-mining/selected.jsonl \
  --stats-output /workspace/data/xzz_data/DAHUA/runtime/hard-mining/class-stats.json \
  --review-db /workspace/data/xzz_data/DAHUA/runtime/visualization/review/review.sqlite3
```

验收 `class-stats.json` 是否自动统计类别频次和稀有类；检查所选样本是否
按熵、Top-1/Top-2 间隔、稳定性、姿态质量、稀有度以及师生冲突排序，
并自动出现在 Web 待审队列。临时参数可用 `--minimum-score`、
`--limit`、`--per-class-limit` 等覆盖 YAML。

## 四、六分类伪标签、蒸馏与增量学习验收

当前 NTU120 联调与最终 Campus6 训练是两个标签空间。不得把 NTU120
Top-5 直接注入六分类训练。恢复六分类骨架数据后，依次执行：

1. 用独立人工校准集运行 `calibrate_teacher`；
2. 运行 `generate_pseudo_dataset`，核对 `accepted/review/rejected`；
3. 在 Web 完成 review，核对版本化 JSONL 回写和撤销；
4. 运行 `build_campus6_dataset`，确认伪标签只进入 train，val/test
   只有人工标签；
5. 用普通配置微调一轮，再用 `--distill` 配置至少跑一个 epoch；
6. 检查日志存在 `loss_hard`、`loss_pseudo`、`loss_qwen_kd`、
   `loss_previous_kd`，并且均为有限值；
7. 构建回放集后再训练一轮，比较旧类、新类、危险类和校准指标；
8. 注册候选、执行发布门禁并演练一次回滚。

完整命令及数据字段见
[`campus6_learning_loop.md`](campus6_learning_loop.md)。

## 五、效果类验收标准

以下项目不能由单元测试替代：

- 在固定人工测试集报告 Macro-F1、每类召回率、混淆矩阵和 ECE；
- 按无遮挡/遮挡、正面/侧面/背面、白天/夜间分别报告指标；
- 人工抽检解释是否由对应证据片段支持，而非只看文字是否流畅；
- 比较增量前后旧类指标，验证灾难性遗忘门禁；
- 对最终部署文件执行真实大小检查并测量目标硬件延迟；
- 保存数据版本、配置哈希、权重哈希、阈值和运行日志，保证可复现。

在这些真实结果产生之前，答辩材料应使用“已实现工程框架、待数据恢复
后完成模型验收”，不要写成“所有指标已经达标”。
