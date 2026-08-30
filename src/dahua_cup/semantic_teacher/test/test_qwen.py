from pathlib import Path

from dahua_cup.semantic_teacher.api.qwen_api import QwenTeacher
from dahua_cup.semantic_teacher.pseudo_label.pseudo_labeler import PseudoLabeler

if __name__ == "__main__":

    teacher = QwenTeacher()
    labeler = PseudoLabeler()

    video_path = str(Path(__file__).resolve().parents[1] / "examples" / "test.mp4")

    # =========================
    # Teacher 推理
    # =========================
    result = teacher.classify_behavior(video_path)

    print(result)

    # =========================
    # 伪标签系统
    # =========================
    labeler.add_sample(video_path, result)
    labeler.save()
