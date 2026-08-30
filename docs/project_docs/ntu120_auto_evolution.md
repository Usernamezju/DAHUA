# NTU120 自动伪标签、蒸馏与安全发布

## 一键启动

服务器上只需启动同一个 Web 服务：

```bash
cd /workspace/code/DAHUA
conda activate skel_gcn38
bash dahua_cup/scripts/run_visualization.sh
```

`run_visualization.sh` 默认启用
`configs/campus/auto_evolution_ntu120.yaml`。服务启动后后台控制器会立即
执行一次扫描，之后默认每 30 秒扫描一次：

1. 合并 Web 产生的 ProtoGCN NTU120 预测和 Qwen 教师结果；
2. 生成持久化的 NTU120 伪标签 JSONL；
3. 把待复核样本写入现有 SQLite 人工审核队列；
4. 按熵、Top-1/Top-2 间隔、姿态质量、时序不稳定、稀有类别和师生冲突
   挖掘难例；
5. 判断是否满足训练条件；
6. 满足时从当前线上权重快照启动 NTU120 logits 蒸馏；
7. 在相同的原始 NTU120 XSub 验证集上比较基线与候选；
8. 仅在全部发布门禁通过后原子切换线上权重。

Web 的“闭环中心”可查看等待条件、伪标签数量、难例数量、当前权重和最近
验证结果，也可点击“立即扫描”。命令行可以覆盖默认配置：

```bash
bash dahua_cup/scripts/run_visualization.sh \
  --evolution-config dahua_cup/configs/campus/auto_evolution_ntu120.yaml \
  --enable-evolution
```

需要只运行推理审核台而暂时停止自动闭环时：

```bash
bash dahua_cup/scripts/run_visualization.sh --disable-evolution
```

## 自动训练条件

默认条件全部满足才训练：

- 至少 20 条已接受且骨架 NPZ 仍然存在的 NTU120 伪标签；
- 相比上一次尝试至少新增 10 条；
- 存在匹配当前 NTU25 Bone 配置的 NTU120 3D XSub 标注；
- 存在 ProtoGCN 训练环境和当前线上 checkpoint；
- Web 设置中已选择足量且当前可见的训练 GPU；
- 距离上次训练尝试至少 12 小时。

默认查找以下任一基础标注：

```text
/workspace/data/xzz_data/DAHUA/datasets/NTU/ProtoGCN/ntu120_3danno.pkl
/workspace/data/xzz_data/DAHUA/datasets/ntu120_3d/ntu120_3danno.pkl
```

如果缺少标注或条件不足，服务仍正常运行，只在闭环状态中记录明确等待
原因，不会启动空训练。门槛、轮询周期、epoch、学习率和回放容量都可在
YAML 中调整。

## 伪标签与人工复核

NTU120 伪标签的联合质量分由以下信号组成：

```text
0.45 × Qwen confidence
+ 0.20 × 学生/教师 Top-1 一致性
+ 0.20 × 骨架质量
+ 0.15 × 多次学生推理稳定性
```

Qwen 当前的 `confidence` 是模型在结构化输出中自报的未校准数值，因此
记录中明确标为 `qwen_self_report_uncalibrated`，不能解释为真实准确率。
训练只接受达到门槛、无强制复核原因且骨架特征存在的记录。师生高置信
冲突、教师主动要求复核或处在中间分数区间的样本写入 SQLite 队列。

当前 Web 人工按钮是 Campus6 六分类审核，不能把该人工六分类结果直接
当成 NTU120 硬标签。NTU120 自动训练只使用通过筛选的 120 类教师软分布；
队列中的人工结果作为审计和后续 Campus6 数据来源保留。

## 权重保护与发布

每次训练前只读取当前生产指针并使用其 checkpoint 初始化；训练过程中
在线推理继续使用旧权重。候选导出为 weight-only checkpoint 后必须同时
满足：

- `mean_class_accuracy` 至少提升 0.001；
- Top-1 相对基线下降不超过 0.002；
- checkpoint 不超过 50 MiB。

失败候选进入归档，生产指针不变。通过候选依次进入
`validated → canary → production`，生产指针原子更新；后续新推理任务
自动读取新 checkpoint，已经运行中的任务不受影响。
第一次训练验证时，系统也会把启动前的原始 checkpoint 注册为 baseline，
因此第一次自动升级后仍可通过生产指针回滚到原始权重。

## 产物与状态

```text
runtime/visualization/evolution/
├── state.json
├── pseudo/ntu120_pseudo_labels.jsonl
├── pseudo/ntu120_pseudo_history.jsonl
├── hard_mining/selected.jsonl
├── hard_mining/class_stats.json
├── models/registry.jsonl
├── models/production.json
└── training/<UTC-run-id>/
    ├── ntu120_distill.pkl
    ├── dataset_build.log
    ├── training.log
    ├── candidate_weights.pth
    └── release_report.json
```

可直接检查 API：

```bash
curl -fsSL http://127.0.0.1:8000/api/evolution
curl -fsSL -X POST http://127.0.0.1:8000/api/evolution/run
```

完整代码级回归：

```bash
python -m dahua_cup.scripts.verify_requirements
```

真实训练仍需要服务器上的 NTU120 标注、模型权重、CUDA 和 ProtoGCN
依赖；代码测试不会伪造“训练指标已经提升”。
