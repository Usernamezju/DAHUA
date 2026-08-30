import json

import numpy as np
import pytest

from dahua_cup.feature_extraction.mediapipe_pose import (
    assign_pose_slots,
    mediapipe33_to_ntu25,
    normalized_pose_center,
)
from dahua_cup.pipeline.protogcn_student_worker import (
    DEFAULT_LABEL_MAP,
    build_student_evidence,
    load_labels,
    load_pose_feature,
    previous_inference_history,
)


def test_mediapipe33_to_ntu25_maps_direct_and_virtual_joints():
    landmarks = np.arange(99, dtype=np.float32).reshape(33, 3)
    visibility = np.ones(33, dtype=np.float32)
    mapped, scores, valid = mediapipe33_to_ntu25(landmarks, visibility)

    assert mapped.shape == (25, 3)
    assert np.allclose(mapped[0], landmarks[[23, 24]].mean(axis=0))
    assert np.allclose(mapped[1], landmarks[[23, 24, 11, 12]].mean(axis=0))
    assert np.allclose(mapped[4], landmarks[11])
    assert np.allclose(mapped[24], landmarks[22])
    assert np.allclose(scores, 1.0)
    assert valid.all()


def test_mediapipe_mapping_zeros_low_visibility_and_non_finite_joints():
    landmarks = np.ones((33, 3), dtype=np.float32)
    visibility = np.ones(33, dtype=np.float32)
    visibility[11] = 0.0
    landmarks[22, 0] = np.nan

    mapped, _, valid = mediapipe33_to_ntu25(
        landmarks, visibility, score_threshold=0.75
    )

    assert not valid[4]
    assert np.all(mapped[4] == 0)
    assert not valid[24]
    assert np.all(mapped[24] == 0)


def test_pose_slots_start_left_to_right_then_preserve_identity():
    first = np.asarray([[0.8, 0.5], [0.2, 0.5]], dtype=np.float32)
    assert assign_pose_slots(first, max_people=2) == [1, 0]

    previous = [first[1], first[0]]
    second = np.asarray([[0.75, 0.5], [0.25, 0.5]], dtype=np.float32)
    assert assign_pose_slots(second, previous, max_people=2) == [1, 0]


def test_normalized_pose_center_uses_hips():
    landmarks = np.zeros((33, 3), dtype=np.float32)
    landmarks[23, :2] = [0.2, 0.4]
    landmarks[24, :2] = [0.6, 0.8]
    assert np.allclose(normalized_pose_center(landmarks), [0.4, 0.6])


def test_load_pose_feature_contract(tmp_path):
    path = tmp_path / "pose.npz"
    keypoint = np.ones((2, 7, 25, 3), dtype=np.float32)
    np.savez_compressed(
        path,
        schema_version=np.asarray("mediapipe_ntu25.v1"),
        keypoint=keypoint,
    )
    assert np.array_equal(load_pose_feature(path), keypoint)

    empty_path = tmp_path / "empty.npz"
    np.savez_compressed(
        empty_path,
        schema_version=np.asarray("mediapipe_ntu25.v1"),
        keypoint=np.zeros_like(keypoint),
    )
    with pytest.raises(ValueError, match="no usable pose"):
        load_pose_feature(empty_path)


def test_load_ntu120_labels():
    labels = load_labels(DEFAULT_LABEL_MAP)
    assert len(labels) == 120
    assert labels[0] == "drink water"
    assert labels[-1] == "rock-paper-scissors"


def test_student_worker_preserves_bounded_inference_history(tmp_path):
    output = tmp_path / "prediction.json"
    output.write_text(
        json.dumps(
            {
                "generated_at": "run-1",
                "topk": [
                    {"label": "walking", "score": 0.8},
                    {"label": "running", "score": 0.1},
                ],
                "inference_history": [
                    {
                        "generated_at": f"old-{index}",
                        "top1_label": "walking",
                        "top1_score": 0.7,
                    }
                    for index in range(5)
                ],
            }
        ),
        encoding="utf-8",
    )
    history = previous_inference_history(output)
    assert len(history) == 4
    assert history[-1]["generated_at"] == "run-1"
    assert history[-1]["top1_top2_margin"] == pytest.approx(0.7)


def test_student_evidence_uses_measurements_and_reports_limitations(tmp_path):
    feature = tmp_path / "pose.npz"
    valid = np.zeros((2, 4, 25), dtype=bool)
    valid[0, :2, :] = True
    points = np.zeros((2, 4, 25, 2), dtype=np.float32)
    points[0, :, :, 0] = np.arange(4, dtype=np.float32)[:, None]
    np.savez_compressed(
        feature,
        schema_version=np.asarray("mediapipe_ntu25.v1"),
        keypoint=np.ones((2, 4, 25, 3), dtype=np.float32),
        valid_mask=valid,
        image_keypoint=points,
        fps=np.asarray(2.0),
        width=np.asarray(10.0),
        height=np.asarray(10.0),
    )
    evidence = build_student_evidence(
        "sample",
        feature,
        [
            {"label": "run on the spot", "score": 0.60},
            {"label": "walking towards", "score": 0.20},
        ],
    )
    assert evidence["method"] == "semantic_graph.measured_pose_and_student_margin"
    assert evidence["evidence"][0]["top1_top2_margin"] == pytest.approx(0.40)
    assert evidence["evidence"][1]["active_person_count"] == 1
    assert evidence["evidence"][1]["pose_coverage"] == pytest.approx(0.5)
    assert any(item["code"] == "low_pose_coverage" for item in evidence["limitations"])
    assert any(item["code"] == "limited_interaction_evidence" for item in evidence["limitations"])
