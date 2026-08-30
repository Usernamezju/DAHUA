# 算法测试指南

## 1. 代码级验收

```bash
cd /workspace/code/DAHUA

/root/miniconda3/envs/skel_gcn38/bin/python -m \
  dahua_cup.scripts.verify_requirements \
  --output \
  /workspace/data/xzz_data/DAHUA/runtime/requirements-code.json
```

这一步只验证代码、CLI 和控制逻辑，不代表模型精度通过。

## 2. Web 与外部资产验收

运行前先检查 GPU；不要根据历史状态选择卡号：

```bash
nvidia-smi
```

Web 启动后执行：

```bash
cd /workspace/code/DAHUA

/root/miniconda3/envs/skel_gcn38/bin/python -m \
  dahua_cup.scripts.verify_requirements \
  --runtime \
  --data-root /workspace/data/xzz_data/DAHUA \
  --base-url http://127.0.0.1:8000 \
  --qwen-model-dir /workspace/data/public_data/Qwen3-VL-8B-Instruct \
  --qwen-dtype float16 \
  --mediapipe-model \
  /workspace/data/xzz_data/DAHUA/models/pose_models/mediapipe/pose_landmarker_heavy.task \
  --edge-model \
  /workspace/data/xzz_data/DAHUA/models/pose_models/mediapipe/pose_landmarker_heavy.task \
  --edge-model \
  /workspace/data/xzz_data/DAHUA/models/protogcn_pretrained/ntu120_xsub_bone_b1/best_top1_acc_epoch_150.pth \
  --output \
  /workspace/data/xzz_data/DAHUA/runtime/requirements-runtime.json \
  2>&1 | tee \
  /workspace/data/xzz_data/DAHUA/runtime/logs/requirements-runtime.log
```

使用 Heavy 时完整包预期会在 50MiB 门禁失败，这是正确的审计结果。下载并
验证 Lite 后，把两个 Heavy 路径替换成 `pose_landmarker_lite.task`。

## 3. 固定人工测试集格式

评测 JSONL 每行代表一个人工真值样本：

```json
{"sample_id":"clip-001","true_label":"normal_walk","predicted_label":"normal_walk","distribution":{"normal_walk":0.82,"normal_run":0.08,"playful_chase":0.03,"playful_push":0.02,"conflict_chase":0.03,"conflict_push":0.02},"latency_ms":18.4,"slices":{"occlusion":"none","viewpoint":"front","lighting":"day"}}
```

要求：

- `sample_id` 唯一；
- `true_label` 只能来自固定六类；
- 提供完整 `distribution` 才能计算 Brier；仅有置信度仍可计算 ECE；
- `latency_ms` 必须覆盖完整端到端推理，而不是只测 GCN forward；
- `occlusion`、`viewpoint`、`lighting` 由人工按固定规范标注。

## 4. 生成正式指标与 Markdown 报告

```bash
cd /workspace/code/DAHUA

mkdir -p \
  /workspace/data/xzz_data/DAHUA/experiments/acceptance/campus6 \
  /workspace/data/xzz_data/DAHUA/runtime/logs

/root/miniconda3/envs/skel_gcn38/bin/python -u -m \
  dahua_cup.pipeline.evaluate_predictions \
  --input-jsonl \
  /workspace/data/xzz_data/DAHUA/datasets/campus6/test_predictions.jsonl \
  --output \
  /workspace/data/xzz_data/DAHUA/experiments/acceptance/campus6/metrics.json \
  --markdown-output \
  /workspace/data/xzz_data/DAHUA/experiments/acceptance/campus6/test_report.md \
  --ece-bins 15 \
  --slice-field occlusion \
  --slice-field viewpoint \
  --slice-field lighting \
  --edge-model \
  /workspace/data/xzz_data/DAHUA/models/pose_models/mediapipe/pose_landmarker_lite.task \
  --edge-model \
  /workspace/data/xzz_data/DAHUA/models/protogcn_campus6/best.pth \
  --model-size-limit-mib 50 \
  2>&1 | tee \
  /workspace/data/xzz_data/DAHUA/runtime/logs/campus6_acceptance.log
```

## 5. 必须执行的对照实验

1. MediaPipe Heavy 与 Lite：相同视频、相同 GCN、相同阈值。
2. Student only 与 Student→Qwen 门控：报告准确率、Qwen 调用率与平均时延。
3. 人工标签 baseline 与加入伪标签/蒸馏：报告每类指标和 ECE。
4. 增量前与增量后：分别报告旧数据、新数据、危险类，验证遗忘门禁。
5. 常规集与遮挡/视角/光照分组：不得只报告总体平均数。

每次报告必须保存数据版本、代码 commit、配置、checkpoint SHA-256、模型包
文件列表、GPU/CPU 型号和完整日志。
