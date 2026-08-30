import json

from dahua_cup.pipeline.extract_dataset import main as extract_main
from dahua_cup.pipeline.generate_pseudo_dataset import main as pseudo_main
from dahua_cup.pipeline.run_closed_loop import main as closed_loop_main
from dahua_cup.pipeline.train_student import main as train_main
from dahua_cup.semantic_teacher.schemas import LABELS, TeacherOutput


def distribution(label, probability=0.99):
    rest = (1 - probability) / (len(LABELS) - 1)
    return {item: probability if item == label else rest for item in LABELS}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_extract_and_train_dry_run(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    config = tmp_path / "config.yaml"
    config.write_text("student: {}\n", encoding="utf-8")
    manifest = tmp_path / "videos.jsonl"
    write_jsonl(manifest, [{"sample_id": "s1", "video": str(video)}])
    extract_main(["--config", str(config), "--input-manifest", str(manifest),
                  "--output-dir", str(tmp_path / "out"), "--dry-run"])
    train_main(["--config", str(config), "--dry-run"])


def test_generate_pseudo_dataset_dry_run(tmp_path):
    output = TeacherOutput("s1", "normal_walk", distribution("normal_walk"), 0.99)
    teacher_path, student_path = tmp_path / "teacher.jsonl", tmp_path / "student.jsonl"
    write_jsonl(teacher_path, [output.to_dict()])
    write_jsonl(student_path, [{"sample_id": "s1", "distribution": distribution("normal_walk"),
                                "pose_quality": 0.99, "temporal_stability": 0.99}])
    pseudo_main(["--teacher-jsonl", str(teacher_path), "--student-jsonl", str(student_path),
                 "--output-dir", str(tmp_path / "pseudo"), "--dataset-version", "v1",
                 "--prompt-version", "p1", "--dry-run"])


def test_closed_loop_dry_run(tmp_path):
    manifest = tmp_path / "batch.jsonl"
    write_jsonl(manifest, [{"sample_id": "s1"}])
    closed_loop_main(["--base-model", "m1", "--new-data", str(manifest),
                      "--output-manifest", str(tmp_path / "run.json"),
                      "--stage-command", "extract=python -V", "--dry-run"])
