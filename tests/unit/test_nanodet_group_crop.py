import numpy as np
import pytest

from dahua_cup.person_localization.grouping import (
    PersonDetection,
    build_group_crop,
    crop_and_letterbox,
    select_people,
)
from dahua_cup.person_localization.nanodet_backend import (
    NanoDetPersonDetector,
)


def person(box, confidence):
    return PersonDetection(tuple(map(float, box)), float(confidence))


def test_zero_people_falls_back_to_full_frame():
    assert build_group_crop([], width=640, height=360) == (
        0.0,
        0.0,
        640.0,
        360.0,
    )


def test_one_person_crop_contains_the_person_and_adds_context():
    detection = person((250, 80, 350, 330), 0.9)
    group = build_group_crop(
        [detection],
        width=640,
        height=360,
        context_factor=1.2,
        minimum_crop_fraction=0.1,
    )

    assert group[0] < 250
    assert group[1] <= 80
    assert group[2] > 350
    assert group[3] >= 330


def test_two_people_are_merged_by_fixed_geometry():
    first = person((40, 60, 180, 330), 0.95)
    second = person((380, 70, 540, 325), 0.90)
    group = build_group_crop(
        [first, second],
        width=640,
        height=360,
        context_factor=1.1,
        minimum_crop_fraction=0.1,
    )

    assert group[0] <= 40
    assert group[1] <= 60
    assert group[2] >= 540
    assert group[3] >= 330


def test_selection_keeps_two_best_people_and_reports_crowd_overflow():
    selected, overflow = select_people(
        [
            person((0, 0, 20, 50), 0.7),
            person((30, 0, 50, 50), 0.95),
            person((60, 0, 80, 50), 0.8),
            person((90, 0, 110, 50), 0.2),
        ],
        width=120,
        height=60,
        confidence_threshold=0.35,
        maximum_people=2,
    )

    assert [item.confidence for item in selected] == [0.95, 0.8]
    assert overflow == 1


def test_letterbox_preserves_crop_aspect_ratio():
    pytest.importorskip("cv2")
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    output, transform = crop_and_letterbox(
        frame, (0, 0, 200, 100), output_size=100
    )

    assert output.shape == (100, 100, 3)
    assert transform["resized_width"] == 100
    assert transform["resized_height"] == 50
    assert transform["offset_y"] == 25


def test_nanodet_output_parser_keeps_continuous_person_boxes():
    result = {
        0: np.asarray(
            [[1.5, 2.5, 30.25, 50.75, 0.91]], dtype=np.float32
        )
    }

    detections = NanoDetPersonDetector.parse_person_results(result, 0)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.91)
    assert detections[0].bbox_xyxy == pytest.approx(
        (1.5, 2.5, 30.25, 50.75)
    )


def test_nanodet_selector_unwraps_official_image_id_mapping():
    official_result = {
        0: {
            0: [[1.0, 2.0, 30.0, 50.0, 0.91]],
            1: [],
        }
    }

    per_image = NanoDetPersonDetector.select_image_results(
        official_result, image_id=0
    )
    detections = NanoDetPersonDetector.parse_person_results(per_image, 0)

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.91)


def test_nanodet_selector_accepts_sequence_wrappers():
    per_image = NanoDetPersonDetector.select_image_results(
        [{0: [[1.0, 2.0, 30.0, 50.0, 0.91]]}]
    )

    assert 0 in per_image


@pytest.mark.parametrize("result", [[], {}, None, np.zeros((1, 5))])
def test_nanodet_selector_rejects_empty_or_unsupported_results(result):
    with pytest.raises(RuntimeError):
        NanoDetPersonDetector.select_image_results(result)


def test_worker_parser_defaults_to_no_temporal_smoothing():
    from dahua_cup.pipeline.nanodet_group_crop_worker import build_parser

    args = build_parser().parse_args(
        [
            "--video",
            "input.mp4",
            "--output",
            "crop.mp4",
            "--metadata",
            "crop.json",
        ]
    )

    assert not hasattr(args, "smoothing")
    assert args.crowd_policy == "full-frame"
