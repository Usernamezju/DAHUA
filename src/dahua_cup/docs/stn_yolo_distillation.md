# YOLO蒸馏的集合感知群组STN

## 目标和边界

该子系统只解决“小人物在送入姿态模型前如何自适应裁剪和放大”。
训练教师是冻结的YOLO人物检测/跟踪模型，训练过程中不调用MediaPipe、
ProtoGCN或Qwen。

部署链路为：

```text
原视频 -> Set-aware Group STN -> 放大视频 -> MediaPipe -> ProtoGCN
```

模型内部最多预测两个person query，但最终始终只输出一个群组裁剪框：

- 没有人：使用全画面；
- 一个人：使用单人框及上下文；
- 两个人：使用两个框的联合区域及上下文。

当前三人以上的训练视频默认跳过。纯人物检测教师无法判断三人以上场景中
哪两个人在语义上构成关键交互。

## 网络结构

`dahua_cup/stn/model.py`实现：

1. RGB加固定XY坐标通道；
2. 轻量深度可分离3D卷积骨干；
3. 多尺度FPN，保留小人物特征；
4. 两个person queries和集合预测头；
5. 时序卷积细化逐帧人物轨迹；
6. 训练辅助人物占用图；
7. 可微分人物框联合和方形上下文扩张；
8. 仅含等比例缩放和平移的Spatial Transformer。

默认模型有61,718个可训练参数。其仿射矩阵固定为：

```text
[[s, 0, tx],
 [0, s, ty]]
```

网络无法表达旋转、剪切、非等比例缩放或时间轴变换。

## 第一步：用YOLO生成教师记录

安装教师导出依赖时，应在已有CUDA版PyTorch环境中安装Ultralytics，
不要让`pip`覆盖服务器现有的CUDA版PyTorch。

```bash
python -m pip install ultralytics opencv-python
```

建议使用精度优先的大型YOLO人物检测权重。权重只在离线导出阶段使用，
不会进入最终部署包。

```bash
python -m dahua_cup.stn.export_yolo_teacher \
  --input-manifest \
    /workspace/data/xzz_data/DAHUA/datasets/vedio/manual_label_manifest.csv \
  --output \
    /workspace/data/xzz_data/DAHUA/datasets/stn/yolo_tracks.jsonl \
  --model /workspace/data/xzz_data/DAHUA/models/yolo/yolo11x.pt \
  --device 0 \
  --image-size 1280 \
  --confidence 0.50 \
  --crowd-policy skip
```

先检查输入而不加载YOLO：

```bash
python -m dahua_cup.stn.export_yolo_teacher \
  --input-manifest \
    /workspace/data/xzz_data/DAHUA/datasets/vedio/manual_label_manifest.csv \
  --output \
    /workspace/data/xzz_data/DAHUA/datasets/stn/yolo_tracks.jsonl \
  --model /workspace/data/xzz_data/DAHUA/models/yolo/yolo11x.pt \
  --dry-run
```

每个视频完成后，接受记录和处理状态立即保存在输出路径对应的`.parts/`
目录；全部分片成功后再合并最终JSONL。教师记录包含：

- 视频路径和原始尺寸；
- 每帧最多两个人物框；
- YOLO置信度；
- 跟踪ID；
- 短时漏检插值标记；
- 教师模型和阈值。

### 多GPU教师导出

教师导出按视频并行，而不是拆分单个视频内部的连续帧。开始前先检查所有
GPU的显存、利用率和占用进程：

```bash
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv
```

确认0～7号GPU均可用后，使用8个独立YOLO进程：

```bash
python -m dahua_cup.stn.export_yolo_teacher \
  --input-manifest \
    /workspace/data/xzz_data/DAHUA/datasets/vedio/manual_label_manifest.csv \
  --output \
    /workspace/data/xzz_data/DAHUA/datasets/stn/yolo_tracks.jsonl \
  --model /workspace/data/xzz_data/DAHUA/models/yolo/yolo11x.pt \
  --devices 0,1,2,3,4,5,6,7 \
  --image-size 1280 \
  --confidence 0.50 \
  --crowd-policy top2
```

每张GPU处理一个确定的视频分片。接受记录和处理状态逐视频写入
`yolo_tracks.jsonl.parts/`，所有进程成功后才按原manifest顺序合并最终
JSONL。任务中断后使用完全相同的命令会跳过已经处理的视频并继续；若教师
模型、阈值、跟踪器或manifest发生变化，必须使用新的`--output`路径，避免
混合不兼容的伪标注。

如果明确决定放弃尚未处理的视频，可以将已有接受记录定稿为部分教师集：

```bash
python -m dahua_cup.stn.export_yolo_teacher \
  --input-manifest \
    /workspace/data/xzz_data/DAHUA/datasets/vedio/manual_label_manifest.csv \
  --output \
    /workspace/data/xzz_data/DAHUA/datasets/stn/yolo_tracks.jsonl \
  --model /workspace/data/xzz_data/DAHUA/models/yolo/yolo11x.pt \
  --device 0 \
  --image-size 1280 \
  --confidence 0.50 \
  --crowd-policy top2 \
  --finalize-partial
```

该操作不会把剩余视频记为已处理；相邻的
`yolo_tracks.jsonl.summary.json`会明确记录`partial`和`unprocessed`，
原`.parts/`仍可用于以后续跑。

## 第二步：只训练STN

默认配置：

```text
dahua_cup/configs/stn/yolo_distillation.yaml
```

运行：

```bash
python -m dahua_cup.stn.train \
  --config dahua_cup/configs/stn/yolo_distillation.yaml \
  --teacher-jsonl \
    /workspace/data/xzz_data/DAHUA/datasets/stn/yolo_tracks.jsonl \
  --output-dir \
    /workspace/data/xzz_data/DAHUA/models/stn/set_aware_group \
  --device cuda:0
```

单卡可通过`--device cuda:4`选择；多卡可通过
`--devices 4,5,6,7`启用PyTorch DataParallel。多卡列表第一张是主卡，
检查点始终保存未包装模型的权重，因此单卡和多卡训练可以相互恢复：

```bash
python -m dahua_cup.stn.train \
  --config dahua_cup/configs/stn/yolo_distillation.yaml \
  --teacher-jsonl \
    /workspace/data/xzz_data/DAHUA/datasets/stn/yolo_tracks.jsonl \
  --output-dir \
    /workspace/data/xzz_data/DAHUA/models/stn/set_aware_group \
  --devices 4,5,6,7
```

命令行参数会覆盖YAML。训练自动使用`1.0、0.5、0.25`三档整体缩放：
它会同时缩小画面和YOLO框，并随机改变缩小画面在大画布中的位置。
默认采样器同时按来源数据集和人数分层加权，避免数量更多的KTH单人视频
淹没BEHAVE/LIMU双人视频。

输出包括：

```text
best.pt
latest.pt
training_report.json
```

训练损失由presence、单人框L1/GIoU、群组框、全人物包含约束、人物占用图
和时序轨迹监督组成。最多两个查询时只比较两种排列，实现精确集合匹配，
不依赖SciPy。

## 独立可视化检验台

STN检验台与校园行为主工作台相互独立。它不会修改主Web的推理流程，
也不会自动把STN接到MediaPipe前面。服务读取训练输出的`best.pt`和YOLO
教师JSONL，为所选视频生成：

- 两个人物查询框、最终联合框与YOLO教师框的叠加视频；
- STN实际裁剪放大后的视频；
- 联合框IoU、人物完整包含率、人数判断准确率、时序抖动和放大倍率；
- IoU最低的帧索引，便于定位远距离、遮挡或双人分离等失败模式。

教师记录中的`crowd_frame_ratio`会作为容量告警展示。只要视频中存在三人
以上的帧，页面就明确说明IoU、包含率和人数指标只覆盖教师选中的top-2，
不能用这些指标证明模型已经处理了其余人物。

开始前先检查所有GPU的显存和利用率，再显式选择空闲卡。默认服务端口为
`8010`，不会占用主工作台的`8000`端口：

```bash
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv
```

例如使用4号GPU：

```bash
bash dahua_cup/scripts/run_stn_visualization.sh \
  --device \
  cuda:4
```

完整的可覆盖参数入口为：

```bash
python -m dahua_cup.stn.visualization_app \
  --checkpoint \
  /workspace/data/xzz_data/DAHUA/models/stn/set_aware_group/best.pt \
  --teacher-jsonl \
  /workspace/data/xzz_data/DAHUA/datasets/stn/yolo_tracks.jsonl \
  --output-dir \
  /workspace/data/xzz_data/DAHUA/runtime/stn_visualization \
  --device \
  cuda:4 \
  --batch-size \
  4 \
  --window-step \
  8 \
  --host \
  0.0.0.0 \
  --port \
  8010
```

浏览器访问`http://服务器地址:8010`。第一次选择样本运行时才会加载
PyTorch检查点并生成视频；相同检查点和样本的结果会缓存。勾选
“忽略缓存重新生成”可强制重新推理。

## KTH、BEHAVE、LIMU的使用

三套视频可以一起作为第一版YOLO蒸馏语料，但作用不同：

- KTH：主要提供单人、行走/慢跑/奔跑/拳击和尺度变化；
- BEHAVE：主要提供WalkTogether、RunTogether、Chase、Fight双人轨迹；
- LIMU：提供push、punch、pull、kick、hug、touch、handshake等近距离双人
  交互。

它们不能绕过YOLO教师导出直接训练，因为现有统一CSV只有视频路径和来源
行为标签，没有逐帧人物框。STN不使用这些行为类别，只使用YOLO生成的人物
框、presence、置信度和轨迹。

这三套数据适合完成冷启动和功能验证，但不应作为最终唯一训练集。它们规模
较小、场景较旧且与真实校园监控存在域差异。最终至少需要加入一批无人工
标注的校园固定摄像头视频，再由同一个YOLO教师离线生成框。

验证集必须按人物或原始长视频分组，而不能把同一人物/同一长视频切出的片段
随机分到训练集和验证集。

## 原始平移坐标

当前ProtoGCN使用MediaPipe世界坐标的Bone模态，本身不使用人物在原图中的
全局平移。Qwen语义图则会从`image_keypoint`计算轨迹、速度和双人距离。

部署STN时必须保存每帧`theta`及letterbox几何参数，并将裁剪画面中的
MediaPipe图像关键点逆映射回原视频坐标。不能把裁剪后的局部像素坐标直接
写入现有`image_keypoint`字段。
