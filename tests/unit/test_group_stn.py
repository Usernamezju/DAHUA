import importlib.util
import json
from pathlib import Path

import pytest

from dahua_cup.stn.export_yolo_teacher import (
    _export_summary,
    merge_part_records,
    parse_devices,
    shard_sources,
)
from dahua_cup.stn.teacher_labels import (
    SCHEMA_VERSION,
    interpolate_track_gaps,
    select_track_ids,
    validate_teacher_record,
)
from dahua_cup.stn.train import parse_cuda_devices


def _person(track_id, left, confidence=0.9):
    return {
        "track_id": track_id,
        "bbox_xyxy": [left, 0.2, left + 0.2, 0.8],
        "confidence": confidence,
        "interpolated": False,
    }


def test_yolo_track_selection_prefers_persistent_people():
    frames = [
        {"frame_index": 0, "persons": [_person(1, 0.1), _person(2, 0.6)]},
        {"frame_index": 1, "persons": [_person(1, 0.1), _person(2, 0.6)]},
        {
            "frame_index": 2,
            "persons": [
                _person(1, 0.1),
                _person(2, 0.6),
                _person(9, 0.4, 0.99),
            ],
        },
    ]
    assert select_track_ids(
        frames, maximum_people=2, minimum_coverage=0.5
    ) == [1, 2]


def test_short_yolo_track_gap_is_interpolated():
    frames = [
        {"frame_index": 0, "persons": [_person(3, 0.1, 0.8)]},
        {"frame_index": 1, "persons": []},
        {"frame_index": 2, "persons": [_person(3, 0.5, 0.6)]},
    ]
    output = interpolate_track_gaps(frames, [3], maximum_gap=1)
    middle = output[1]["persons"][0]
    assert middle["interpolated"]
    assert middle["confidence"] == pytest.approx(0.6)
    assert middle["bbox_xyxy"][0] == pytest.approx(0.3)


def test_teacher_record_contract_accepts_one_or_two_people(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": "clip",
        "video_path": str(video),
        "selected_track_ids": [1, 2],
        "frames": [
            {
                "frame_index": 0,
                "persons": [_person(1, 0.1), _person(2, 0.6)],
            }
        ],
    }
    validate_teacher_record(record)


def test_multi_gpu_device_parser_and_balanced_video_shards():
    assert parse_devices("0, 2,7") == [0, 2, 7]
    with pytest.raises(ValueError, match="duplicate"):
        parse_devices("0,0")
    sources = [{"sample_id": str(index)} for index in range(8)]
    shards = shard_sources(sources, 3)
    assert [len(shard) for shard in shards] == [3, 3, 2]
    assert [row["sample_id"] for row in shards[0]] == ["0", "3", "6"]


def test_stn_training_cuda_device_parser():
    assert parse_cuda_devices("4, 5,7") == [4, 5, 7]
    assert parse_cuda_devices([0, 2]) == [0, 2]
    with pytest.raises(ValueError, match="duplicate"):
        parse_cuda_devices("4,4")


def test_multi_gpu_part_merge_is_manifest_ordered_and_deduplicated(tmp_path):
    first_video = tmp_path / "first.mp4"
    second_video = tmp_path / "second.mp4"
    first_video.write_bytes(b"first")
    second_video.write_bytes(b"second")

    def record(sample_id, video_path, track_id):
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "video_path": str(video_path),
            "selected_track_ids": [track_id],
            "frames": [
                {
                    "frame_index": 0,
                    "persons": [_person(track_id, 0.1)],
                }
            ],
        }

    part_zero = tmp_path / "accepted-worker-00.jsonl"
    part_one = tmp_path / "accepted-worker-01.jsonl"
    first_record = record("first", first_video, 1)
    second_record = record("second", second_video, 2)
    part_zero.write_text(
        "\n".join(
            [
                json.dumps(second_record),
                json.dumps(second_record),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    part_one.write_text(
        json.dumps(first_record) + "\n",
        encoding="utf-8",
    )
    sources = [
        {"sample_id": "first", "video_path": Path(first_video)},
        {"sample_id": "second", "video_path": Path(second_video)},
    ]
    merged = merge_part_records([part_zero, part_one], sources)
    assert [row["sample_id"] for row in merged] == ["first", "second"]


def test_partial_export_summary_keeps_pending_videos_unprocessed(tmp_path):
    sources = [
        {"sample_id": "accepted", "video_path": tmp_path / "accepted.mp4"},
        {"sample_id": "rejected", "video_path": tmp_path / "rejected.mp4"},
        {"sample_id": "pending", "video_path": tmp_path / "pending.mp4"},
    ]
    accepted = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": "accepted",
        "video_path": str(sources[0]["video_path"]),
        "selected_track_ids": [1],
        "frames": [
            {
                "frame_index": 0,
                "persons": [_person(1, 0.1)],
            }
        ],
    }
    progress = {
        str(sources[0]["video_path"].resolve()): {
            "status": "accepted",
        },
        str(sources[1]["video_path"].resolve()): {
            "status": "no_persistent_person",
        },
    }
    summary = _export_summary(
        sources,
        progress,
        [accepted],
        [0],
        tmp_path / "teacher.jsonl",
        tmp_path / "teacher.jsonl.parts",
        partial=True,
    )
    assert summary["processed"] == 2
    assert summary["accepted"] == 1
    assert summary["unprocessed"] == 1
    assert summary["partial"]
    assert summary["rejected"] == {"no_persistent_person": 1}


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is installed only in the server training environment",
)
def test_group_stn_shapes_rotation_prior_and_backward():
    import torch

    from dahua_cup.stn.losses import GroupSTNDistillationLoss
    from dahua_cup.stn.model import (
        SetAwareGroupSTN,
        group_boxes_from_queries,
        trainable_parameter_count,
    )

    boxes = torch.tensor(
        [[[[0.1, 0.2, 0.3, 0.8], [0.6, 0.1, 0.9, 0.7]]]]
    )
    both = group_boxes_from_queries(
        boxes, torch.tensor([[[10.0, 10.0]]]), hard=True
    )
    one = group_boxes_from_queries(
        boxes, torch.tensor([[[10.0, -10.0]]]), hard=True
    )
    # The left edge of both crops can clamp to zero; the top edge remains a
    # stable indicator that merging two people expands the group crop.
    assert both[0, 0, 1] < one[0, 0, 1]
    assert both[0, 0, 2] > one[0, 0, 2]

    model = SetAwareGroupSTN(feature_channels=32)
    model.train()
    video = torch.randn(2, 3, 4, 64, 64)
    outputs = model(video, apply_transform=False)
    assert outputs["boxes_xyxy"].shape == (2, 4, 2, 4)
    assert outputs["presence_logits"].shape == (2, 4, 2)
    assert outputs["theta"].shape == (2, 4, 2, 3)
    assert torch.count_nonzero(outputs["theta"][..., 0, 1]) == 0
    assert torch.count_nonzero(outputs["theta"][..., 1, 0]) == 0
    assert torch.allclose(
        outputs["theta"][..., 0, 0], outputs["theta"][..., 1, 1]
    )
    assert trainable_parameter_count(model) < 250_000

    targets = {
        "boxes_xyxy": boxes.expand(2, 4, 2, 4).clone(),
        "presence": torch.ones(2, 4, 2, dtype=torch.bool),
        "confidence": torch.ones(2, 4, 2),
    }
    losses = GroupSTNDistillationLoss()(outputs, targets)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(
        parameter.grad is not None for parameter in model.parameters()
    )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="PyTorch is installed only in the server training environment",
)
def test_group_box_merging_preserves_float16_dtype():
    import torch

    from dahua_cup.stn.losses import target_group_boxes
    from dahua_cup.stn.model import group_boxes_from_queries

    boxes = torch.tensor(
        [[[[0.1, 0.2, 0.3, 0.8], [0.6, 0.1, 0.9, 0.7]]]],
        dtype=torch.float16,
    )
    presence = torch.tensor([[[True, False]]])

    teacher_group = target_group_boxes(
        boxes,
        presence,
        context_factor=1.25,
        minimum_fraction=0.25,
    )
    predicted_group = group_boxes_from_queries(
        boxes,
        torch.tensor([[[10.0, -10.0]]], dtype=torch.float16),
        hard=True,
    )

    assert teacher_group.dtype == torch.float16
    assert predicted_group.dtype == torch.float16
    assert torch.isfinite(teacher_group).all()
    assert torch.isfinite(predicted_group).all()
