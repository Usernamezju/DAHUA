# 校园行为语义理解原型用户手册

## 1. 启动前检查

服务器代码根目录为 `/workspace/code/DAHUA`，数据根目录为
`/workspace/data/xzz_data/DAHUA`。任何 GPU 推理或训练开始前先执行：

```bash
nvidia-smi
```

确认目标 GPU 有足够显存且没有他人任务后再设置 `CUDA_VISIBLE_DEVICES`。

## 2. 启动 Web、NTU120 闭环和 8B FP16 教师

以下示例使用物理 GPU 7。若 GPU 7 不空闲，应根据 `nvidia-smi` 修改。

```bash
cd /workspace/code/DAHUA

mkdir -p \
  /workspace/data/xzz_data/DAHUA/runtime/logs

export CUDA_VISIBLE_DEVICES=7
export DAHUA_VIS_PYTHON=/root/miniconda3/envs/skel_gcn38/bin/python
export DAHUA_TEACHER_PYTHON=/workspace/code/envs/llm_env/bin/python

bash dahua_cup/scripts/run_visualization.sh \
  --disable-nanodet \
  --enable-evolution \
  --qwen-model-dir /workspace/data/public_data/Qwen3-VL-8B-Instruct \
  --qwen-dtype float16 \
  --qwen-device-map auto \
  --qwen-attention sdpa \
  --qwen-max-frames 8 \
  --qwen-max-new-tokens 512 \
  --qwen-retries 2 \
  2>&1 | tee \
  /workspace/data/xzz_data/DAHUA/runtime/logs/visualization_qwen8b_fp16.log
```

浏览器访问 `http://服务器地址:8000`。

## 3. 推理审核台

1. 从下拉框选择服务器视频，或上传授权视频。
2. “生成预览”只处理浏览器兼容格式。
3. “提取骨架”生成 NTU25 特征和 pose-only 视频。
4. “运行完整推理”先运行 ProtoGCN；低置信、小间隔或不稳定样本按配置调用
   Qwen，骨架质量是风险信号而不是直接阻断条件。
5. 查看 Student Top-5、教师建议、置信度和理由。
6. 对需要复核的样本提交六分类人工结果；特殊结果可选无法判断、视频损坏、
   无关样本。

## 4. 主要产物

```text
runtime/visualization/artifacts/features/      NTU25 NPZ
runtime/visualization/artifacts/pose_videos/   pose-only MP4
runtime/visualization/artifacts/predictions/   ProtoGCN JSON
runtime/visualization/artifacts/teachers/      Qwen JSON
runtime/visualization/review/review.sqlite3    审核与事件
runtime/visualization/evolution/               伪标签、难例、训练和模型状态
```

教师 JSON 的 `provenance` 应记录实际 `model_dir`、`torch_dtype`、
`device_map`、Prompt 版本和视频抽帧信息。

## 5. 停止与重启

先确认没有正在写入的教师或训练任务，再执行：

```bash
pkill -TERM -f '[d]ahua_cup.backend.app' || true
pkill -TERM -f '[q]wen_teacher_worker' || true
```

随后重新执行第 2 节命令。历史 feature、prediction、teacher、SQLite 和进化
状态均位于数据根目录，不会因正常重启而删除。

## 6. 常见状态

- `student_confidence_below_threshold`：学生置信度触发教师，不等于预测必错。
- `student_margin_below_threshold`：Top-1/Top-2 过近，属于难例信号。
- `pose_quality_below_threshold`：骨架质量较弱，会提高复核优先级，但可用骨架
  仍继续学生和教师推理。
- `insufficient_accepted_pseudo_labels`：伪标签数量未达到自动训练门槛。
- `training_cooldown`：距离上次训练尝试尚未超过冷却期。
- `waiting Campus6 checkpoint`：六分类模型尚未训练，当前在线学生仍是 NTU120。

代码级与真实指标验收见 [算法测试指南](algorithm_test_guide.md)。
