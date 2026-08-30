from dahua_cup.backend.gpu import GPUManager


def gpu(index, *, used=10, utilization=0):
    return {
        "index": index,
        "name": "RTX 3090",
        "uuid": f"GPU-{index}",
        "memory_total_mb": 24576,
        "memory_used_mb": used,
        "utilization_gpu_percent": utilization,
        "temperature_c": 40,
    }


def test_qwen_auto_and_manual_candidate_selection(tmp_path, monkeypatch):
    manager = GPUManager(tmp_path / "gpus.json")
    snapshots = [gpu(0, used=20), gpu(1, used=10), gpu(2, used=20000)]
    monkeypatch.setattr(manager, "discover", lambda: [dict(item) for item in snapshots])

    automatic = manager.update([0, 1], [0, 1], [], teacher_auto=True)
    assert automatic["teacher_mode"] == "auto"
    assert automatic["teacher_admission"]["gpu_ids"] == [1]

    manual = manager.update([0], [1], [0], teacher_auto=False)
    assert manual["teacher_mode"] == "manual"
    assert manual["teacher_admission"]["gpu_ids"] == [0]


def test_qwen_service_occupancy_is_visible(tmp_path, monkeypatch):
    manager = GPUManager(tmp_path / "gpus.json")
    monkeypatch.setattr(manager, "discover", lambda: [gpu(0)])
    manager.set_teacher_active([0])
    state = manager.state()
    assert state["gpus"][0]["service_busy"] is True
    assert state["gpus"][0]["service_roles"] == ["Qwen3-VL-8B"]
