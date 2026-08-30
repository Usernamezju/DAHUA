import json

import pytest

from dahua_cup.backend.gpu import GPUManager, parse_nvidia_smi
from dahua_cup.backend.jobs import (
    pose_quality_metrics,
    set_command_option,
    teacher_gate_decision,
    teacher_student_conflict,
)


NVIDIA_SMI_OUTPUT = """0, NVIDIA A100-SXM4-80GB, GPU-a, 81920, 1024, 12, 41
1, NVIDIA A100-SXM4-80GB, GPU-b, 81920, 2048, 23, 44
"""


def test_parse_nvidia_smi_inventory():
    gpus = parse_nvidia_smi(NVIDIA_SMI_OUTPUT)
    assert [gpu["index"] for gpu in gpus] == [0, 1]
    assert gpus[1]["memory_used_mb"] == 2048
    assert gpus[0]["temperature_c"] == 41


def test_gpu_selection_is_persisted_and_round_robin(tmp_path, monkeypatch):
    path = tmp_path / "settings" / "gpu.json"
    manager = GPUManager(path)
    monkeypatch.setattr(manager, "discover", lambda: parse_nvidia_smi(NVIDIA_SMI_OUTPUT))

    state = manager.update([], [0, 1])
    assert state["pose_gpu_ids"] == []
    assert state["mediapipe_device"] == "cpu"
    assert json.loads(path.read_text(encoding="utf-8"))["student_gpu_ids"] == [0, 1]

    restored = GPUManager(path)
    assert restored.next_device("student") == 0
    assert restored.next_device("student") == 1


def test_gpu_selection_allows_cpu_pose_and_rejects_empty_student_pool(tmp_path, monkeypatch):
    manager = GPUManager(tmp_path / "gpu.json")
    monkeypatch.setattr(manager, "discover", lambda: parse_nvidia_smi(NVIDIA_SMI_OUTPUT))
    assert manager.update([], [0])["mediapipe_delegate"] == "CPU"
    with pytest.raises(ValueError, match="ProtoGCN"):
        manager.update([], [])


def test_gpu_selection_rejects_invisible_device(tmp_path, monkeypatch):
    manager = GPUManager(tmp_path / "gpu.json")
    monkeypatch.setattr(manager, "discover", lambda: parse_nvidia_smi(NVIDIA_SMI_OUTPUT))
    with pytest.raises(ValueError, match="GPU 不存在"):
        manager.update([], [7])


def test_command_device_is_forcibly_replaced():
    command = ["python", "worker.py", "--device", "cuda:7"]
    assert set_command_option(command, "--device", "cuda:0")[-2:] == [
        "--device", "cuda:0"
    ]
    assert set_command_option(["python", "worker.py"], "--delegate", "gpu")[-2:] == [
        "--delegate", "gpu"
    ]


def test_gpu_process_environment_maps_physical_card_to_logical_zero():
    environment = GPUManager.process_environment(7)
    assert environment["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert environment["CUDA_VISIBLE_DEVICES"] == "7"
    assert environment["DAHUA_PHYSICAL_GPU_ID"] == "7"


def test_teacher_gate_uses_ntu120_top1_confidence():
    high = teacher_gate_decision(
        {
            "task": "ntu120_xsub",
            "topk": [
                {"label": "walking", "score": 0.80},
                {"label": "running", "score": 0.20},
            ],
        },
        0.70,
    )
    assert not high["triggered"]
    assert high["route"] == "use_student"
    assert high["reason"] == "student_result_reliable"

    low = teacher_gate_decision(
        {
            "task": "ntu120_xsub",
            "topk": [
                {"label": "walking", "score": 0.42},
                {"label": "running", "score": 0.30},
            ],
        },
        0.70,
    )
    assert low["triggered"]
    assert low["student_confidence"] == 0.42

    missing = teacher_gate_decision(
        {"task": "ntu120_xsub", "topk": []}, 0.70
    )
    assert missing["triggered"]
    assert missing["reason"] == "invalid_student_top1_confidence"


def test_teacher_gate_routes_small_margin_instability_and_bad_pose():
    prediction = {
        "topk": [
            {"label": "walking", "score": 0.82},
            {"label": "running", "score": 0.75},
        ]
    }
    margin = teacher_gate_decision(prediction, 0.70, margin_threshold=0.15)
    assert margin["route"] == "call_teacher"
    assert "student_margin_below_threshold" in margin["reasons"]

    unstable = teacher_gate_decision(
        prediction,
        0.70,
        margin_threshold=0.05,
        instability_score=0.90,
    )
    assert unstable["route"] == "call_teacher"
    assert unstable["hard_sample"]

    bad_pose = teacher_gate_decision(
        prediction,
        0.70,
        pose_quality=0.40,
        pose_quality_threshold=0.70,
    )
    assert bad_pose["route"] == "call_teacher"
    assert bad_pose["triggered"]
    assert "pose_quality_below_threshold" in bad_pose["reasons"]


def test_pose_quality_excludes_empty_person_slot(tmp_path):
    import numpy as np

    feature = tmp_path / "pose.npz"
    valid = np.zeros((2, 4, 25), dtype=bool)
    valid[0] = True
    scores = np.zeros((2, 4, 25), dtype=np.float32)
    scores[0] = 0.9
    np.savez_compressed(
        feature, valid_mask=valid, keypoint_score=scores
    )
    quality = pose_quality_metrics(feature)
    assert quality["active_people"] == 1
    assert quality["pose_coverage"] == 1.0
    assert quality["quality"] == pytest.approx(0.97)


def test_high_confidence_student_teacher_conflict():
    conflict = teacher_student_conflict(
        {
            "topk": [
                {"label": "walking", "score": 0.88},
                {"label": "running", "score": 0.05},
            ]
        },
        {"result": {"label": "running", "confidence": 0.91}},
        0.70,
    )
    assert conflict["conflict"]
    assert conflict["reason"] == "high_confidence_student_teacher_conflict"
