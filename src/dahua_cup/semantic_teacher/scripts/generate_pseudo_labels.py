import json
import os
from pathlib import Path

from dahua_cup.semantic_teacher.api.qwen_api import QwenTeacher
print("generate_pseudo_labels.py started")

# ==========================
# 配置
# ==========================

# 先使用 test 目录，后面比赛直接改成 dataset/train 即可
COMPETITION_ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = Path(
    os.environ.get(
        "DAHUA_TEACHER_VIDEO_DIR",
        COMPETITION_ROOT / "semantic_teacher" / "examples",
    )
).expanduser()

OUTPUT_DIR = Path(
    os.environ.get(
        "DAHUA_TEACHER_OUTPUT_DIR",
        COMPETITION_ROOT / "semantic_teacher" / "pseudo_labels",
    )
).expanduser()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ACCEPTED_FILE = OUTPUT_DIR / "accepted.jsonl"
REVIEW_FILE = OUTPUT_DIR / "review.jsonl"

# 高置信阈值（后面可以调整）
CONFIDENCE_THRESHOLD = 0.90


# ==========================
# 主程序
# ==========================

teacher = QwenTeacher()

video_list = sorted(VIDEO_DIR.glob("*.mp4"))

print("=" * 60)
print(f"Found {len(video_list)} videos.")
print("=" * 60)

accepted_count = 0
review_count = 0

with open(ACCEPTED_FILE, "a", encoding="utf-8") as accepted_fp, \
     open(REVIEW_FILE, "a", encoding="utf-8") as review_fp:

    for idx, video_path in enumerate(video_list, start=1):

        print(f"[{idx}/{len(video_list)}] Processing: {video_path.name}")

        try:

            result = teacher.classify_behavior(str(video_path))

            # 保存视频名称
            result["video"] = video_path.name

            # 保存json字符串
            line = json.dumps(result, ensure_ascii=False)

            if result["confidence"] >= CONFIDENCE_THRESHOLD:

                accepted_fp.write(line + "\n")
                accepted_count += 1
                print("  -> Accepted")

            else:

                review_fp.write(line + "\n")
                review_count += 1
                print("  -> Review")

        except Exception as e:

            print(f"  ERROR: {e}")

print("=" * 60)
print("Finished.")
print(f"Accepted : {accepted_count}")
print(f"Review    : {review_count}")
print("=" * 60)
