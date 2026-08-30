import os
import json
from datetime import datetime
from pathlib import Path


DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[1] / "dataset"


class PseudoLabeler:

    def __init__(self, save_dir=None):
        self.save_dir = str(Path(save_dir) if save_dir else DEFAULT_DATASET_DIR)

        self.accepted_path = os.path.join(
            self.save_dir, "accepted/pseudo_labels.json"
        )
        self.review_path = os.path.join(
            self.save_dir, "review/review_queue.json"
        )

        os.makedirs(os.path.dirname(self.accepted_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.review_path), exist_ok=True)

        self.accepted_data = []
        self.review_data = []

    # =========================
    # 核心：接收 teacher 输出
    # =========================
    def add_sample(self, video_path, result):

        sample = {
            "video": video_path,
            "label": result.get("label", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "reason": result.get("reason", ""),
            "timestamp": datetime.now().isoformat()
        }

        conf = sample["confidence"]

        # =========================
        # 过滤策略（比赛关键）
        # =========================
        if conf >= 0.75:
            self.accepted_data.append(sample)

        elif conf >= 0.5:
            self.review_data.append(sample)

        else:
            pass  # 丢弃低质量样本

    # =========================
    # 保存数据
    # =========================
    def save(self):

        with open(self.accepted_path, "w", encoding="utf-8") as f:
            json.dump(self.accepted_data, f, ensure_ascii=False, indent=2)

        with open(self.review_path, "w", encoding="utf-8") as f:
            json.dump(self.review_data, f, ensure_ascii=False, indent=2)

        print(f"[OK] Saved:")
        print(f"  accepted: {len(self.accepted_data)}")
        print(f"  review: {len(self.review_data)}")
