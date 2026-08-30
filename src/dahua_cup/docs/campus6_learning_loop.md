# Campus6 六分类微调与增量学习闭环

本文档对应仓库中已经实现的代码路径。当前还没有 Campus6 骨架数据和训练
环境，因此“代码与接口已实现”不等于“模型已经训练并达到发布指标”。

## 一、数据约定

六类及其固定索引为：

| 索引 | 类别 |
|---:|---|
| 0 | `normal_walk` |
| 1 | `normal_run` |
| 2 | `playful_chase` |
| 3 | `playful_push` |
| 4 | `conflict_chase` |
| 5 | `conflict_push` |

骨架特征必须是 NPZ 文件中的 `keypoint`，形状为 `[M,T,25,3]`，`M<=2`。
这与项目 MediaPipe 到 NTU25 的特征输出一致。验证集和测试集只允许人工
真值，自动伪标签只会进入训练集；同一 `source_hash` 跨数据划分会直接
报错。

基础样本清单 JSONL 至少包含：

```json
{"sample_id":"clip-1","feature_path":"/data/clip-1.npz","label":"normal_walk","label_source":"human","split":"train","source_hash":"video-sha256"}
```

## 二、预训练 ProtoGCN 转为六分类

NTU120 的分类头是 120 类，不能把完整权重直接加载到 6 类头。先删除
分类头张量并规范化分布式训练产生的 `module.` 前缀：

```bash
python -m dahua_cup.scripts.prepare_protogcn_campus6_checkpoint \
  --input /data/checkpoints/ntu120_bone.pth \
  --output /data/checkpoints/protogcn_campus6_init.pth
```

构建版本化训练标注：

```bash
python -m dahua_cup.pipeline.build_campus6_dataset \
  --sample-jsonl /data/manifests/campus6.jsonl \
  --output-dir /data/datasets/campus6 \
  --dataset-version campus6-v1
```

生成的版本目录包括 `annotations.pkl`、`manifest.jsonl` 和带 SHA-256 的
`metadata.json`。六分类微调：

```bash
python -m dahua_cup.pipeline.train_campus6 \
  --ann-file /data/datasets/campus6/campus6-v1/annotations.pkl \
  --init-checkpoint /data/checkpoints/protogcn_campus6_init.pth \
  --work-dir /data/experiments/campus6-v1 \
  --gpus 1
```

配置位于 `configs/protogcn/campus6_ntu25_bone.py`：NTU25 Bone 输入、
6 类 `SimpleHead`、低学习率微调骨干且分类头使用 10 倍学习率。

## 三、置信度校准、阈值筛选与人工复核

校准集必须与训练/测试集隔离，每行包含教师六类分布和人工真值
`true_label`：

```bash
python -m dahua_cup.pipeline.calibrate_teacher \
  --input-jsonl /data/calibration/human_teacher.jsonl \
  --output /data/calibration/teacher_temperature.json
```

这一步用人工数据拟合温度参数。没有校准文件时，流水线会明确记录
`uncalibrated_teacher_distribution`，不会把教师自行写出的 `0.9`
包装成已经校准的真实准确率。

先按信息价值挖掘高价值样本。分数综合学生熵、Top-1/Top-2 间隔、时序
不稳定、骨架失败、新颖性、稀有类；教师结果存在时还加入师生分布分歧
和规则冲突：

```bash
python -m dahua_cup.pipeline.mine_candidates \
  --student-jsonl /data/runs/student.jsonl \
  --teacher-jsonl /data/runs/teacher.jsonl \
  --output /data/runs/high_value.jsonl \
  --limit 2000 --per-class-limit 400 --minimum-score 0.25
```

将筛选结果生成版本化伪标签，并把 `review` 样本自动写入 Web 使用的
SQLite 队列：

```bash
python -m dahua_cup.pipeline.generate_pseudo_dataset \
  --teacher-jsonl /data/runs/teacher.jsonl \
  --student-jsonl /data/runs/high_value.jsonl \
  --output-dir /data/pseudo \
  --dataset-version pseudo-v1 \
  --prompt-version campus-prompt-v1 \
  --threshold-config dahua_cup/configs/campus/thresholds.yaml \
  --calibration /data/calibration/teacher_temperature.json \
  --review-db /data/web/review.sqlite3
```

分流结果是 `accepted/review/rejected`。YAML 中的阈值和权重是运行时真实
配置；高置信接受样本按稳定哈希抽样审计，稀有类按独立比例抽审。
Web 提交六分类结果后，JSONL 会回写成人工 one-hot 标签，同时保留原始
教师软标签；特殊标签会置为拒绝。撤销操作也会恢复上一个版本，数据库
保留审核及事件历史。

## 四、伪标签注入与知识蒸馏

新一轮数据构建可以同时注入人工样本、已接受伪标签、回放样本和旧模型
分布：

```bash
python -m dahua_cup.pipeline.build_campus6_dataset \
  --sample-jsonl /data/manifests/campus6.jsonl \
  --pseudo-jsonl /data/pseudo/pseudo-v1/pseudo_labels.jsonl \
  --replay-jsonl /data/replay/replay-v1.jsonl \
  --previous-jsonl /data/runs/previous_model.jsonl \
  --output-dir /data/datasets/campus6 \
  --dataset-version campus6-v2 \
  --parent-version campus6-v1
```

伪标签记录里只要有 `feature_path`，即使基础清单中没有该样本，也会作为
训练样本真正注入。蒸馏训练：

```bash
python -m dahua_cup.pipeline.train_campus6 \
  --ann-file /data/datasets/campus6/campus6-v2/annotations.pkl \
  --init-checkpoint /data/checkpoints/protogcn_campus6_init.pth \
  --work-dir /data/experiments/campus6-v2-distill \
  --distill --gpus 1
```

`DistillRecognizerGCN` 使用五部分目标：

- 人工标签交叉熵；
- 质量分数加权的伪标签交叉熵；
- Qwen 六类软分布与学生 logits 的温度 KL 蒸馏；
- 旧模型分布的 KL 保持，用于降低遗忘；
- 仅在人工作为硬标签的样本上计算 ProtoGCN CSC 损失。

无教师分布或无旧模型分布的行由显式 mask 排除，不会用全零占位产生
虚假蒸馏梯度。

## 五、增量回放、验证、发布与回滚

从历史人工样本按类别、场景、相机和难度确定性分层采样：

```bash
python -m dahua_cup.pipeline.build_replay_manifest \
  --input-jsonl /data/manifests/history.jsonl \
  --output /data/replay/replay-v1.jsonl \
  --capacity 2000
```

训练完成后，将候选模型及其数据、配置、权重哈希登记到追加式注册表：

```bash
python -m dahua_cup.pipeline.register_candidate \
  --model-id protogcn-campus6-v2 \
  --dataset-id campus6-v2 \
  --checkpoint /data/experiments/campus6-v2-distill/best.pth \
  --config dahua_cup/configs/protogcn/campus6_ntu25_bone_distill.py \
  --metrics /data/metrics/candidate.json \
  --latency-ms 18.2 \
  --registry /data/models/registry.jsonl
```

候选指标 JSON 必须包含 `global_macro_f1`、`new_macro_f1`、
`old_macro_f1`、`dangerous_recall`、`ece` 和 `edge_size_bytes`。
与线上基线比较并在全部门禁通过后发布：

```bash
python -m dahua_cup.pipeline.validate_candidate \
  --candidate-metrics /data/metrics/candidate.json \
  --baseline-metrics /data/metrics/production.json \
  --output /data/metrics/release-report.json \
  --model-id protogcn-campus6-v2 \
  --model-registry /data/models/registry.jsonl \
  --production-pointer /data/models/production.json \
  --promote
```

默认门禁禁止总体和新数据 Macro-F1 下降、旧数据 Macro-F1 遗忘超过
0.02、危险行为召回率下降、ECE 增加超过 0.02，以及边端模型超过
50 MiB。生产指针原子更新并保留前一版本；需要回滚时：

```bash
python -m dahua_cup.pipeline.rollback_model \
  --production-pointer /data/models/production.json \
  --actor operator --reason "online regression"
```

## 六、当前可验证边界

仓库测试覆盖阈值配置、温度校准、高价值排序、Web 入队、审核回写与
撤销、伪标签真实注入、数据泄漏保护、预训练头剥离、发布门禁和生产
回滚。拿到数据后仍必须完成三件事：用独立人工集拟合校准参数、实际
训练并选择 checkpoint、在固定旧/新/危险行为测试集上生成发布指标。
