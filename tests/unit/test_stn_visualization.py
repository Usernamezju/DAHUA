import json
from pathlib import Path

import numpy as np
import pytest

from dahua_cup.stn.inference import (
    evaluate_predictions,
    letterbox_geometry,
    letterbox_to_original,
    original_to_letterbox,
    teacher_record_fingerprint,
)
from dahua_cup.stn.teacher_labels import SCHEMA_VERSION


def test_letterbox_box_round_trip_for_landscape_video():
    geometry = letterbox_geometry(640, 360, 128)
    source = np.asarray((0.1, 0.2, 0.7, 0.9), dtype=np.float32)
    restored = letterbox_to_original(
        original_to_letterbox(source, geometry), geometry
    )
    assert np.allclose(restored, source, atol=1e-5)


def test_stn_metrics_are_perfect_for_matching_group_and_presence():
    geometry = letterbox_geometry(100, 100, 128)
    teacher_frames = [
        {
            "frame_index": 0,
            "persons": [
                {
                    "track_id": 1,
                    "bbox_xyxy": [0.25, 0.2, 0.75, 0.8],
                    "confidence": 0.95,
                    "interpolated": False,
                }
            ],
        }
    ]
    group_boxes = np.asarray(
        [[0.2, 0.2, 0.8, 0.8]], dtype=np.float32
    )
    presence = np.asarray([[0.99, 0.01]], dtype=np.float32)

    metrics, frames = evaluate_predictions(
        group_boxes,
        presence,
        teacher_frames,
        geometry,
        context_factor=1.0,
        minimum_fraction=0.25,
        presence_threshold=0.5,
    )

    assert metrics["mean_group_iou"] == pytest.approx(1.0)
    assert metrics["containment_recall"] == pytest.approx(1.0)
    assert metrics["presence_accuracy"] == pytest.approx(1.0)
    assert metrics["temporal_jitter"] == pytest.approx(0.0)
    assert frames[0]["predicted_people"] == 1


def test_teacher_record_fingerprint_changes_with_boxes():
    first = {"frames": [{"bbox": [0.1, 0.2, 0.3, 0.4]}]}
    second = {"frames": [{"bbox": [0.2, 0.2, 0.3, 0.4]}]}

    assert teacher_record_fingerprint(first) != teacher_record_fingerprint(
        second
    )


def test_standalone_stn_web_lists_teacher_samples(tmp_path):
    pytest.importorskip("fastapi")

    from dahua_cup.stn.visualization_app import (
        STNVisualizationSettings,
        VisualizationManager,
        create_app,
    )

    video = tmp_path / "sample.mp4"
    video.write_bytes(b"not-decoded-by-list-endpoint")
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": "sample",
        "video_path": str(video),
        "source_dataset": "TEST",
        "selected_track_ids": [1],
        "frames": [
            {
                "frame_index": 0,
                "persons": [
                    {
                        "track_id": 1,
                        "bbox_xyxy": [0.1, 0.2, 0.5, 0.9],
                        "confidence": 0.9,
                        "interpolated": False,
                    }
                ],
            }
        ],
    }
    teacher_jsonl = tmp_path / "teacher.jsonl"
    teacher_jsonl.write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"lazy-loading-checkpoint")
    frontend = (
        Path(__file__).resolve().parents[2] / "stn" / "web"
    )
    settings = STNVisualizationSettings(
        checkpoint=checkpoint,
        teacher_jsonl=teacher_jsonl,
        output_root=tmp_path / "output",
        frontend_root=frontend,
        device="cpu",
    )

    manager = VisualizationManager(settings)
    try:
        samples = manager.list_samples(
            query=None,
            dataset=None,
            people=None,
            limit=100,
            offset=0,
        )
    finally:
        manager.shutdown()
    app = create_app(settings)
    routes = {route.path for route in app.routes}

    assert samples["items"][0]["sample_id"] == "sample"
    assert samples["items"][0]["crowd_frame_ratio"] == 0.0
    assert not samples["items"][0]["capacity_exceeded"]
    assert "/api/health" in routes
    assert "/api/samples/{sample_id}/run" in routes
    assert "/api/samples/{sample_id}/media/{kind}" in routes
    assert "独立模型检验台" in (
        frontend / "index.html"
    ).read_text(encoding="utf-8")
