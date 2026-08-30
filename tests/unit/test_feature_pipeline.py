import json

import numpy as np

from dahua_cup.feature_extraction.feature_aggregator import FeatureAggregator
from dahua_cup.feature_extraction.graph_builder import build_coco20, build_protogcn_views
from dahua_cup.feature_extraction.keypoint_filter import interpolate_short_gaps
from dahua_cup.feature_extraction.schemas import FeatureMetadata, save_feature_artifact
from dahua_cup.feature_extraction.semantic_graph import build_semantic_graph


def _person(track_id, offset=0.0):
    points = np.arange(34, dtype=np.float32).reshape(17, 2) + offset
    return {"track_id": track_id, "keypoint": points, "keypoint_score": np.ones(17),
            "bbox": np.array([0, 0, 100, 200], dtype=np.float32)}


def test_aggregator_and_artifact_roundtrip(tmp_path):
    arrays, quality = FeatureAggregator(max_people=2).aggregate(
        [[_person(8)], [_person(8, 1), _person(3)], [_person(8, 2), _person(3, 1)]],
        [0.0, 0.1, 0.2],
    )
    assert arrays["keypoint"].shape == (2, 3, 17, 2)
    assert arrays["track_id"].tolist() == [8, 3]
    assert np.allclose(arrays["velocity"][0, 1], [10, 10])
    metadata = FeatureMetadata("sample", "source", 10, 100, 200, 0, 300, quality=quality)
    npz_path, json_path = save_feature_artifact(tmp_path / "sample", arrays, metadata)
    assert set(np.load(npz_path).files) >= {"keypoint", "valid_mask", "acceleration"}
    assert json.loads(json_path.read_text())["schema_version"] == "feature.v1"


def test_coco20_order_score_and_mask_propagation():
    points = np.zeros((1, 1, 17, 2), dtype=np.float32)
    points[..., 11, :] = [2, 4]
    points[..., 12, :] = [6, 8]
    points[..., 5, :] = [2, 0]
    points[..., 6, :] = [4, 2]
    scores = np.ones((1, 1, 17), dtype=np.float32)
    valid = np.ones_like(scores, dtype=bool)
    valid[..., 12] = False
    converted, converted_scores, converted_valid = build_coco20(points, scores, valid)
    assert np.allclose(converted[..., 17, :], [4, 6])
    assert np.allclose(converted[..., 19, :], [3, 1])
    assert np.allclose(converted[..., 18, :], [3.5, 3.5])
    assert not converted_valid[..., 17].item()
    assert not converted_valid[..., 18].item()
    assert converted_scores[..., 17].item() == 0


def test_protogcn_views_have_expected_shape_and_center():
    points = np.ones((1, 2, 17, 2), dtype=np.float32) * 10
    points[..., 5, 1], points[..., 6, 1] = 5, 5
    points[..., 11, 1], points[..., 12, 1] = 15, 15
    score = np.ones((1, 2, 17), dtype=np.float32)
    views = build_protogcn_views(points, score, score > 0, 100, 50)
    assert views["image"].shape == (1, 2, 20, 3)
    assert np.allclose(views["body"][..., 17, :2], 0)


def test_interpolate_only_bounded_gaps():
    points = np.zeros((7, 1, 2), dtype=np.float32)
    points[0, 0] = 0
    points[3, 0] = 3
    points[6, 0] = 6
    valid = np.array([[1], [0], [0], [1], [0], [0], [1]], dtype=bool)
    filled, mask = interpolate_short_gaps(points, valid, max_gap=2)
    assert mask.all()
    assert np.allclose(filled[:, 0, 0], np.arange(7))
    _, long_mask = interpolate_short_gaps(points, valid, max_gap=1)
    assert not long_mask[1, 0]


def test_semantic_graph_keeps_raw_measured_relation():
    center = np.array([[[0, 0], [1, 0], [2, 0]], [[4, 0], [3, 0], [2.2, 0]]], dtype=np.float32)
    velocity = np.gradient(center, axis=1)
    graph = build_semantic_graph("s", [1, 2], center, velocity,
                                 np.ones((2, 3, 17), dtype=bool), [0.0, 1.0, 2.0],
                                 contact_distance=1.0)
    assert graph["schema_version"] == "semantic_graph.v1"
    assert any(relation["type"] == "approach" and "raw" in relation for relation in graph["relations"])
