# NanoDet轻量人物定位

该模块位于`dahua_cup/person_localization/`，与MediaPipe、ProtoGCN和旧的
STN实验代码平级。它调用官方NanoDet实现，只保留`person`类别，并通过固定
几何规则生成MediaPipe使用的联合裁剪。

## 0、1、2人规则

- 0人：使用完整画面，避免生成空裁剪；
- 1人：人物框增加上下文后生成裁剪；
- 2人：先对两个独立人物框求并集，再增加上下文；
- 3人以上：默认回退完整画面，并在JSON中记录`overflow_people`和
  `crowd_frame_ratio`。可以显式设置`--crowd-policy top2`，但这只保证
  包含置信度最高的两个人。

默认逐帧直接使用NanoDet检测结果，不执行时序平滑，也不沿用上一帧框。
因此快速运动不会产生平滑滞后。裁剪结果采用letterbox补边，不会把人物
强制拉伸成正方形。

## 独立调用

官方NanoDet仓库必须已安装到运行该命令的Python环境中，并准备匹配的YAML
配置和checkpoint：

服务器首次部署可一键安装官方`v1.0.0`源码并下载权重。默认选择同为约
0.95M参数、输入分辨率更适合远距离小人物的NanoDet-m-416：

```bash
cd /workspace/code/DAHUA

mkdir -p \
  /workspace/data/xzz_data/DAHUA/runtime/logs

bash dahua_cup/scripts/setup_nanodet.sh \
  --python /root/miniconda3/envs/skel_gcn38/bin/python \
  --data-root /workspace/data/xzz_data/DAHUA \
  2>&1 | tee /workspace/data/xzz_data/DAHUA/runtime/logs/setup_nanodet.log
```

脚本产物为：

- `/workspace/data/xzz_data/DAHUA/models/nanodet/nanodet-m-416.yml`；
- `/workspace/data/xzz_data/DAHUA/models/nanodet/nanodet_m_416.ckpt`；
- `/workspace/data/xzz_data/DAHUA/third_party/nanodet-v1.0.0/`。

如果服务器代理配置不可用，在命令中增加`--no-proxy`。如需更快但对小目标
更弱的320输入版本，增加`--variant nanodet-m`。脚本可重复执行，已存在的
源码与权重会复用，并在结束前以CPU实际加载模型检查配置和权重是否匹配。
脚本默认只安装PyTorch推理依赖，不安装与当前流程无关的ONNX导出工具，因而
不要求服务器额外安装CMake。

```bash
python -m dahua_cup.pipeline.nanodet_group_crop_worker \
  --video /path/to/input.mp4 \
  --output /path/to/group_crop.mp4 \
  --metadata /path/to/group_crop.json \
  --config /path/to/nanodet-m.yml \
  --checkpoint /path/to/nanodet-m.pth \
  --device cpu \
  --confidence 0.35 \
  --context-factor 1.20 \
  --minimum-crop-fraction 0.20 \
  --output-size 640 \
  --crowd-policy full-frame
```

输出JSON保留每帧原始人物框、置信度、联合框、letterbox映射和人数状态，便于
审计一人、两人以及超容量场景。

## 接入主Web

NanoDet联合裁剪默认关闭，即使配置和权重已经安装，MediaPipe仍直接读取
原视频。这样实验性定位不会静默改变主流程的骨架质量。需要再次评估时，
通过参数显式启用：

```bash
bash dahua_cup/scripts/run_visualization.sh \
  --enable-nanodet \
  --nanodet-config /path/to/nanodet-m.yml \
  --nanodet-checkpoint /path/to/nanodet-m.pth \
  --nanodet-python /path/to/python \
  --nanodet-device cpu \
  --nanodet-confidence 0.35 \
  --nanodet-context-factor 1.20 \
  --nanodet-output-size 640 \
  --nanodet-crowd-policy full-frame
```

Web产物保存为：

- `artifacts/localized_videos/<sample_id>.mp4`：送给MediaPipe的视频；
- `artifacts/localization/<sample_id>.json`：逐帧定位审计记录。

要恢复原视频直送MediaPipe的稳定主流程，可显式执行：

```bash
bash dahua_cup/scripts/run_visualization.sh --disable-nanodet
```
