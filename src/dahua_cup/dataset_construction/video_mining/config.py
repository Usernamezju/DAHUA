"""
Campus6 video mining pipeline — configuration.
Downloads videos from YouTube/Bilibili via yt-dlp, slices into clips,
then classifies sampled frames with a hosted multimodal model.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────
DATA_ROOT = Path("/mnt/f/campus6-miner")
DOWNLOADS_DIR = DATA_ROOT / "downloads"  # raw yt-dlp downloads
CLIPS_DIR = DATA_ROOT / "clips"  # sliced short clips
SCREENED_DIR = DATA_ROOT / "screened"  # final classified clips

# ── Gemini API keys (Google AI Studio free tier) ───────────────────
# Set GEMINI_API_KEYS to a comma-separated list in the shell.  Alternatively,
# point GEMINI_API_KEYS_FILE at a local, ignored text file.  The latter accepts
# both a raw key per line and ``name: key`` / ``name：key`` lines.  Do not place
# credentials in this source file or commit them to the repository.
def _load_gemini_api_keys() -> list[str]:
    from_environment = os.environ.get("GEMINI_API_KEYS", "")
    if from_environment.strip():
        return [key.strip() for key in from_environment.split(",") if key.strip()]

    keys_file = os.environ.get("GEMINI_API_KEYS_FILE", "").strip()
    if not keys_file:
        return []
    try:
        lines = Path(keys_file).expanduser().read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise RuntimeError(f"cannot read GEMINI_API_KEYS_FILE: {exc}") from exc

    keys = []
    for line in lines:
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        if "：" in value:
            value = value.rsplit("：", 1)[1].strip()
        elif ":" in value:
            value = value.rsplit(":", 1)[1].strip()
        if value:
            keys.append(value)
    return keys


GEMINI_API_KEYS = _load_gemini_api_keys()
GEMINI_MODEL = "gemini-3.5-flash"

# Rate limits (free tier): 15 RPM / 1500 per day per key
GEMINI_RPM = 14  # per key, leave margin
GEMINI_MAX_DAILY = 1400  # per key, leave margin

# ── ModelScope Qwen3-VL (default classifier) ──────────────────────
# Get a token from https://modelscope.cn/my/myaccesstoken and export it as
# MODELSCOPE_API_KEY.  Multiple comma-separated tokens are supported when
# available, but a single token is enough for the free tier.
CLASSIFIER_BACKEND = os.environ.get("CAMPUS6_CLASSIFIER_BACKEND", "modelscope").strip().lower()
MODELSCOPE_API_KEYS = [
    key.strip()
    for key in os.environ.get("MODELSCOPE_API_KEY", "").split(",")
    if key.strip()
]
MODELSCOPE_BASE_URL = os.environ.get(
    "MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1"
).rstrip("/")
MODELSCOPE_MODEL = os.environ.get(
    "MODELSCOPE_MODEL", "Qwen/Qwen3-VL-8B-Instruct"
)
MODELSCOPE_RPM = int(os.environ.get("MODELSCOPE_RPM", "6"))
MODELSCOPE_QUOTA_BACKOFF_SECONDS = int(
    os.environ.get("MODELSCOPE_QUOTA_BACKOFF_SECONDS", "3600")
)

# ── DashScope Qwen-VL (local process, server credential optional) ─────────
# The key is supplied only through the local process environment.  It is never
# stored in this repository; the local launcher may retrieve it from the
# user's server secret file for the duration of one run.
DASHSCOPE_API_KEYS = [
    key.strip()
    for key in os.environ.get("DASHSCOPE_API_KEY", "").split(",")
    if key.strip()
]
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
).rstrip("/")
DASHSCOPE_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3-vl-flash")
DASHSCOPE_RPM = int(os.environ.get("DASHSCOPE_RPM", "6"))

# Qwen3-VL is given still frames rather than the original video container.
# Eight evenly spaced 512 px frames preserve a five-second interaction while
# keeping free-tier request sizes and latency manageable.
VLM_FRAME_COUNT = int(os.environ.get("CAMPUS6_VLM_FRAME_COUNT", "8"))
VLM_FRAME_MAX_EDGE = int(os.environ.get("CAMPUS6_VLM_FRAME_MAX_EDGE", "512"))

# Proxy for Google APIs (Windows Clash)
PROXY_URL = "http://172.21.224.1:7897"

# ── yt-dlp download settings ───────────────────────────────────────
YTDLP_FORMAT = "mp4"
YTDLP_SEARCH_COUNT = int(os.environ.get("CAMPUS6_YTDLP_SEARCH_COUNT", "50"))
YTDLP_MAX_DOWNLOADS = int(os.environ.get("CAMPUS6_YTDLP_MAX_DOWNLOADS", "25"))
YTDLP_MAX_DURATION = int(os.environ.get("CAMPUS6_YTDLP_MAX_DURATION", "120"))
YTDLP_ARCHIVE_FILE = DATA_ROOT / "yt_dlp_archive.txt"

# ── Slice settings ─────────────────────────────────────────────────
CLIP_DURATION = float(os.environ.get("CAMPUS6_CLIP_DURATION", "5"))
CLIP_OVERLAP = float(os.environ.get("CAMPUS6_CLIP_OVERLAP", "1"))
MIN_CLIP_DURATION = 2.5  # skip clips shorter than this

# ── Keywords — focused on the 4 under-represented classes ──────────
KEYWORDS = {
    "conflict_chase": [
        "two people foot chase after argument",
        "two people chasing each other street altercation",
        "CCTV person chasing another person fight",
        "people running away from fight on foot",
        "human foot chase altercation surveillance",
        "men chasing each other after argument",
        # Chinese
        "两人冲突追逐",
        "人追人街头冲突",
        "追打逃跑两个人",
        "两人追赶打架",
        "街头人追人",
        "监控两人追逐",
    ],
    "conflict_push": [
        "two people arguing and pushing street",
        "two people shoving match surveillance",
        "person pushes another person argument",
        "two men pushing each other street",
        "CCTV two people physical shove",
        "human altercation push shove two people",
        # Chinese
        "两人冲突推搡",
        "两个人推搡打架",
        "街头两人推人",
        "两人动手推搡",
        "监控两人推搡",
    ],
    "playful_chase": [
        "two children playing tag full body",
        "kids chasing each other playground full body",
        "two people playful chase running",
        "children playing chase outdoor full body",
        "kids running tag game two children",
        # Chinese
        "嬉戏追逐",
        "嬉闹追跑",
        "追逐嬉戏",
        "小朋友追跑",
        "玩耍追逐",
        "追逐打闹",
    ],
    "playful_push": [
        "two children playful pushing full body",
        "kids play fighting two children full body",
        "two people friendly pushing playing",
        "children roughhousing two kids full body",
        "kids playful shoving outdoor",
        # Chinese
        "嬉戏推搡",
        "打闹嬉戏",
        "玩闹推搡",
        "嬉闹玩耍",
        "小朋友打闹",
    ],
    "normal_walk": [
        "students walking campus full body",
        "people walking sidewalk full body",
        "single person walking outdoor full body",
        "school children walking playground full body",
        "校园学生走路全身",
        "一个人走路全身",
        "操场学生走路",
        "街头行人走路全身",
    ],
    "normal_run": [
        "students running campus full body",
        "people jogging street full body",
        "single runner running outdoor full body",
        "school children running playground full body",
        "校园跑步全身",
        "一个人跑步全身",
        "操场学生跑步",
        "街头跑步全身",
    ],
}

# ── Campus6 labels ─────────────────────────────────────────────────
CAMPUS6_CLASSES = [
    "normal_walk",
    "normal_run",
    "playful_chase",
    "playful_push",
    "conflict_chase",
    "conflict_push",
    "irrelevant",
]

# ── Gemini classification prompt (reuses existing audit prompts) ───
CLASSIFY_PROMPT = """你是Campus6校园行为视频数据集的严格审核员。观看这段短视频，判断它属于以下哪个类别：

目标标签（7选1）：
1. normal_walk — 正常行走；持续位移，无追逐/冲突。
2. normal_run — 正常奔跑；持续跑步，无追赶。
3. playful_chase — 嬉戏追逐；两人参与，有角色交换/折返/等待，无防御性逃离。
4. playful_push — 嬉戏推搡；两人间轻接触，身体稳定，接触后继续互动。
5. conflict_chase — 冲突追逐；单向持续追赶，被追者逃离/躲避/惊慌加速。
6. conflict_push — 冲突推搡；突然单向推人，被推者失衡/后退/防御。
7. irrelevant — 不属于以上（超两人、动作不完整、画面损坏、无目标动作、单人无互动等）。

严格规则：
- Campus6只收录真实人类行为。车辆、动物（包括狗、熊等）、动画/游戏角色、玩具或机器人不是“人”；主体不是可确认的真实人类时一律判irrelevant。
- 动物追人/追动物、车辆追人/追车、人与动物互动、人与车辆互动均不属于任何目标类；即使有人逃跑也不得判conflict_chase。
- 任何目标标签都要求：在动作发生的关键帧中，至少一名真实人物从头到脚或几乎完整的全身清晰可见。只有脸、手、腿、背部、局部躯干，或人物被画面边缘/遮挡明显截断时，不满足条件，判irrelevant。
- 对playful_chase、playful_push、conflict_chase、conflict_push四类交互行为：两名主要人物都必须是可确认的真实人类，且两人都从头到脚或几乎完整的全身清晰可见；第一视角仅拍到对方、镜头外有人、只出现一人的局部、任一人严重裁切/遮挡，均判irrelevant。
- 四类交互行为必须恰好两名主要人物；拥挤人群、多人围观而无法明确锁定这两名完整人物时判irrelevant。
- 片名、字幕、配音、新闻标题或评论文字不是行为证据；必须依据画面中可见的完整真人动作判断。
- 不得把两人一起跑自动判为追逐
- 不得把身体接触自动判为推搡
- 只有证据明确才给完整标签，否则判irrelevant

只输出一个JSON对象（不要Markdown代码围栏）：
{
  "campus6_label": "<正常7选1>",
  "confidence": "high|medium|low",
  "reason": "<不超过50字的关键证据>"
}"""

# Changing a safety or labelling rule invalidates earlier local classifications.
CLASSIFY_PROMPT_VERSION = "full_person_v3"
