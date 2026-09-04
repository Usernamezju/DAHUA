import pickle

import numpy as np

from dahua_cup.semantic_teacher.incremental.campus_dataset import (
    CampusIncrementalDataset,
)


def _write_annotation(path, annotations, split):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"annotations": annotations, "split": split}, handle)


def _feature(path, frames=4):
    np.savez(
        path,
        keypoint=np.zeros((2, frames, 17, 2), dtype=np.float32),
        keypoint_score=np.ones((2, frames, 17), dtype=np.float32),
        total_frames=np.asarray(frames),
    )


def _baseline(root):
    _write_annotation(
        root / "campus6_baseline" / "annotations_with_all.pkl",
        [{
            "frame_dir": "baseline/0", "total_frames": 4, "label": 0,
            "keypoint": np.zeros((2, 4, 17, 2), dtype=np.float32),
            "keypoint_score": np.ones((2, 4, 17), dtype=np.float32),
            "img_shape": (1, 1),
        }],
        {"train": ["baseline/0"], "val": [], "test": []},
    )


def _approved(sample_id, label):
    return {
        "sample_id": sample_id,
        "manual_label": label,
        "reviewer": "reviewer",
        "incremental_reason": "human_reviewed_hard_sample",
    }


def test_incremental_batch_is_merged_only_after_success(tmp_path):
    dataset_root = tmp_path / "dataset"
    _baseline(dataset_root)
    manager = CampusIncrementalDataset(dataset_root, tmp_path / "runtime", batch_size=2)
    feature_a, feature_b = tmp_path / "a.npz", tmp_path / "b.npz"
    _feature(feature_a); _feature(feature_b)

    manager.stage(_approved("review/a", "normal_walk"), feature_path=feature_a)
    ready = manager.stage(_approved("review/b", "conflict_push"), feature_path=feature_b)
    assert ready["ready"] is True
    assert (dataset_root / "campus_all" / "annotations_with_all.pkl").exists()

    round_ = manager.prepare_round()
    with (dataset_root / "campus_all" / "annotations_with_all.pkl").open("rb") as handle:
        assert len(pickle.load(handle)["annotations"]) == 1

    result = manager.complete_round(round_, training_metadata={"exit_code": 0})
    assert result["merged_sample_count"] == 2
    assert result["staged_sample_count"] == 0
    with (dataset_root / "campus_all" / "annotations_with_all.pkl").open("rb") as handle:
        merged = pickle.load(handle)
    assert len(merged["annotations"]) == 3
    assert {item["frame_dir"] for item in merged["annotations"]} == {
        "baseline/0", "review/a", "review/b"
    }


def test_later_arrivals_are_not_erased_when_completed_batch_is_cleared(tmp_path):
    dataset_root = tmp_path / "dataset"
    _baseline(dataset_root)
    manager = CampusIncrementalDataset(dataset_root, tmp_path / "runtime", batch_size=2)
    paths = [tmp_path / f"{index}.npz" for index in range(3)]
    for path in paths: _feature(path)
    manager.stage(_approved("review/a", "normal_walk"), feature_path=paths[0])
    manager.stage(_approved("review/b", "normal_run"), feature_path=paths[1])
    round_ = manager.prepare_round()
    manager.stage(_approved("review/c", "playful_chase"), feature_path=paths[2])

    manager.complete_round(round_)
    assert manager.status()["staged_sample_count"] == 1
    record = manager.records()[0]
    assert record["sample_id"] == "review/c"
    assert (manager.increment_root / record["feature"]).is_file()


def test_scheduled_round_freezes_exact_batch_and_leaves_later_samples(tmp_path):
    dataset_root = tmp_path / "dataset"
    _baseline(dataset_root)
    manager = CampusIncrementalDataset(dataset_root, tmp_path / "runtime", batch_size=2)
    paths = [tmp_path / f"scheduled-{index}.npz" for index in range(3)]
    for path in paths:
        _feature(path)
    manager.stage(_approved("review/a", "normal_walk"), feature_path=paths[0])
    manager.stage(_approved("review/b", "normal_run"), feature_path=paths[1])
    manager.stage(_approved("review/c", "playful_chase"), feature_path=paths[2])

    round_ = manager.prepare_round()
    assert set(round_.sample_ids) == {"review/a", "review/b"}
    with round_.incoming_annotation.open("rb") as handle:
        incoming = pickle.load(handle)
    assert len(incoming["annotations"]) == 2
    assert sum(len(values) for values in incoming["split"].values()) == 2
    manager.complete_round(round_)
    assert [row["sample_id"] for row in manager.records()] == ["review/c"]
