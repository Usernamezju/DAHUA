# Campus6 数据集构建 — 现状总结

## 项目背景

Campus6 校园行为分类包含六类行为，每类目标 500 个样本：

| 编号 | 标签 | 含义 |
| --- | --- | --- |
| 1 | normal_walk | 正常行走 |
| 2 | normal_run | 正常奔跑 |
| 3 | playful_chase | 嬉戏追逐 |
| 4 | playful_push | 嬉戏推搡 |
| 5 | conflict_chase | 冲突追逐 |
| 6 | conflict_push | 冲突推搡 |

当前极其缺少后四类交互行为样本。

## 一、服务器端：VLM 筛选管道（远程）

服务器为 `10.14.114.133:27596`，运行在 Docker 容器内。核心代码位于
`/workspace/code/DAHUA/dahua_cup/dataset_construction/`：

```text
workflow.py          核心筛选工作流
providers.py         VLM API 客户端（OpenAI 兼容格式）
decision.py          多模型共识决策
screen_dataset.py    入口脚本
contact_sheets.py    视频 → contact sheet
configs/             GLM、Qwen、Gemini 通道配置
prompts/             公共规则和三阶段审核提示词
```

候选清单位于
`/workspace/data/xzz_data/DAHUA/datasets/campus6_candidates/manifests/`：

| 清单 | 数量 | 说明 |
| --- | ---: | --- |
| `multimodal_candidates.jsonl` | 8,867 | 全量 RGB 视频 |
| `unprocessed_qwen_candidates.jsonl` | 7,731 | Qwen 增量 |
| `skeleton_candidates.jsonl` | 7,524 | 骨架数据，已废弃 |

| 通道 | 模型 | 候选数 | 已处理 | 状态 |
| --- | --- | ---: | ---: | --- |
| GLM 原始 | glm-4.6v-flash / glm-4.1v-thinking-flash | 8,867 | 1,136 | tmux `campus6_glm`，刚重启 |
| Qwen 增量 | qwen3-vl-flash / qwen3-vl-plus | 7,731 | 330 | 反复崩溃，未运行 |
| 骨架 Qwen | qwen3-vl-flash | 7,524 | 1,236 | 已废弃；火柴人骨架无法可靠区分意图 |

当前已筛出样本（GLM + Qwen）：

| 类别 | GLM | Qwen | 合计 | 距 500 缺额 |
| --- | ---: | ---: | ---: | ---: |
| normal_walk | 216 | 129 | 345 | 155 |
| normal_run | 38 | 5 | 43 | 457 |
| playful_chase | 7 | 36 | 43 | 457 |
| playful_push | 2 | 37 | 39 | 461 |
| conflict_chase | 3 | 0 | 3 | 497 |
| conflict_push | 1 | 3 | 4 | 496 |

### 已知问题与处理

1. Qwen（GLM 偶发）会在 OpenCV 解码个别视频时因
   `corrupted double-linked list` 或 `free(): invalid next size` 被 glibc 直接终止。
   Python 异常处理不能捕获该类错误。
2. `contact_sheets.py` 已加入每视频子进程隔离：子进程成功完成并写好元数据后才原子提交
   contact sheet；子进程崩溃、异常退出或超时会让主流程为该候选写入 `error` review，随后继续下一个视频。
3. 后四类交互行为的候选池本身不足，仍需以视频挖矿补充。
4. 两个服务器通道使用付费 DashScope API，应持续关注调用成本。

典型重启方式：

```bash
# GLM
tmux new-session -d -s campus6_glm "... hosted_screening.json ..."

# Qwen（修复部署后）
tmux new-session -d -s campus6_qwen "... hosted_screening_qwen.json 和 unprocessed_qwen_candidates.jsonl ..."
```

## 二、本地端：视频挖矿管道（WSL）

代码位于 `dahua_cup/dataset_construction/video_mining/`：

```text
config.py       关键词、环境变量和路径配置
download.py     yt-dlp 搜索与下载
slice_clips.py  ffmpeg 切片
classify.py     ModelScope Qwen3-VL（默认）/ Gemini 分类
run_pipeline.py 主流程入口
```

数据根目录为 `F:\campus6-miner\`（WSL：`/mnt/f/campus6-miner/`），其中：

```text
downloads/      原始下载视频
clips/          5 秒切片
screened/       按六类及 irrelevant 分类的结果
```

已验证 yt-dlp 下载和 ffmpeg 切片。Gemini 的 `generateContent` 目前返回
`403 PERMISSION_DENIED`：`count_tokens` 正常，说明是 Google Cloud 项目访问被拒，
需要在项目中绑定信用卡或改用新的项目和 key，并非单纯地区限制。

默认筛选器已切换为 ModelScope Qwen3-VL：视频会均匀抽取 8 帧，以 OpenAI 兼容接口发送。
获取 ModelScope token 后运行：

```bash
export MODELSCOPE_API_KEY='your_modelscope_token'
python -m dahua_cup.dataset_construction.video_mining.run_pipeline --screen
```

连续挖矿与筛选（下载、切片和推理并发；结果可恢复）：

```bash
export MODELSCOPE_API_KEY='your_modelscope_token'
python -m dahua_cup.dataset_construction.video_mining.continuous \
  --producer-rounds 1 --max-downloads-per-keyword 1
```

`--producer-rounds 0` 会持续抓取；应结合 ModelScope 的实际每日免费额度谨慎使用。

Gemini key 不再写入源码；若需显式回退 Gemini，运行前通过环境变量提供：

```bash
export GEMINI_API_KEYS='key_1,key_2'
export CAMPUS6_CLASSIFIER_BACKEND='gemini'
python -m dahua_cup.dataset_construction.video_mining.run_pipeline --screen
```

也可使用本地 key 文件（每行可为裸 key、`name: key` 或 `name：key`）：

```bash
export GEMINI_API_KEYS_FILE='/path/to/api key.txt'
python -m dahua_cup.dataset_construction.video_mining.run_pipeline --screen
```

运行方式：

```bash
cd ~/projects/behaviour_recognition
conda activate base
python -m dahua_cup.dataset_construction.video_mining.run_pipeline
python -m dahua_cup.dataset_construction.video_mining.run_pipeline --download
```

## 三、后续事项

1. 获取 ModelScope token，并先对一个片段做连通性和标签质量测试。
2. Gemini 若恢复可作为独立复核通道（绑定计费，或在新项目创建 API key）。
3. 将本次 OpenCV 隔离修复同步到服务器后，重启 Qwen 增量通道并观察坏视频是否只导致单条失败而不再杀死 tmux 主进程。
