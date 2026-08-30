import json
from pathlib import Path

from dahua_cup.dataset_construction.decision import (
    decide_reviews,
    parse_model_response,
    validate_review,
)
from dahua_cup.dataset_construction.build_candidate_manifest import (
    enforce_cross_modality_disjoint,
    hmdb51_push_records,
    quota_report,
    uav_human_rgb_records,
)
from dahua_cup.dataset_construction.download_candidate_sources import _kinetics_rows
from dahua_cup.dataset_construction.screen_dataset import main
from dahua_cup.dataset_construction.providers import HostedVisionClient
from dahua_cup.dataset_construction.skeleton_sheets import create_skeleton_contact_sheets
from dahua_cup.dataset_construction.workflow import ScreeningWorkflow, discover_directory


def valid_review(label="playful_chase"):
    components = {
        "normal_walk": ("walk", "neutral", 1),
        "normal_run": ("run", "neutral", 1),
        "playful_chase": ("chase", "playful", 2),
        "playful_push": ("push", "playful", 2),
        "conflict_chase": ("chase", "conflict", 2),
        "conflict_push": ("push", "conflict", 2),
    }
    motion, intent, people = components[label]
    return {
        "primary_person_count": people,
        "third_person_present": False,
        "action_interval_seconds": {"start": 0.2, "end": 2.8},
        "motion_primitive": motion,
        "intent": intent,
        "campus6_label": label,
        "evidence": {
            "unilateral_pursuit": False,
            "role_exchange": True,
            "repeated_return": True,
            "escape_behavior": False,
            "defensive_pose": False,
            "physical_contact": motion == "push",
            "strong_displacement": intent == "conflict" and motion == "push",
            "loss_of_balance": False,
        },
        "person_visibility": "good",
        "action_completeness": "complete",
        "evidence_grade": "A",
        "decision": "accept_full",
        "reason": "两名主要人物动作和结果完整可见",
    }


def test_json_fence_is_parsed_and_validated():
    text = "```json\n" + json.dumps(valid_review(), ensure_ascii=False) + "\n```"
    parsed = parse_model_response(text)
    assert parsed.valid
    assert parsed.label == "playful_chase"


def test_interaction_with_third_person_is_rejected_by_schema():
    value = valid_review("conflict_push")
    value["third_person_present"] = True
    parsed = validate_review(value)
    assert not parsed.valid
    assert "full label cannot contain a persistent third person" in parsed.errors


def test_two_blind_reviews_are_required_for_acceptance():
    first = validate_review(valid_review("playful_push"))
    second = validate_review(valid_review("playful_push"))
    decision = decide_reviews([first, second])
    assert decision == {
        "status": "accepted",
        "label": "playful_push",
        "reason": "two_model_full_consensus",
    }


def test_disagreement_without_adjudicator_goes_to_manual_review():
    first = validate_review(valid_review("playful_chase"))
    second = validate_review(valid_review("conflict_chase"))
    assert decide_reviews([first, second])["status"] == "manual_review"


def test_pose_interaction_requires_three_model_consensus():
    first = validate_review(valid_review("conflict_push"))
    second = validate_review(valid_review("conflict_push"))
    without_third = decide_reviews(
        [first, second],
        require_unanimous_interaction=True,
    )
    with_third = decide_reviews(
        [first, second],
        adjudicator=validate_review(valid_review("conflict_push")),
        require_unanimous_interaction=True,
    )
    assert without_third["status"] == "manual_review"
    assert with_third["status"] == "accepted"
    assert with_third["reason"] == "three_model_pose_interaction_consensus"


def test_directory_discovery_uses_video_extensions(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "clip.mp4").write_bytes(b"candidate")
    (tmp_path / "readme.txt").write_text("ignore", encoding="utf-8")
    candidates = discover_directory(tmp_path, [".mp4"])
    assert len(candidates) == 1
    assert candidates[0]["path"].endswith("clip.mp4")


def test_cli_dry_run_never_needs_an_api_key(tmp_path, monkeypatch):
    input_root = tmp_path / "candidates"
    output_root = tmp_path / "screened"
    input_root.mkdir()
    (input_root / "clip.mp4").write_bytes(b"candidate")
    monkeypatch.delenv("ZHIPUAI_API_KEY", raising=False)
    code = main(
        [
            "--input",
            str(input_root),
            "--output",
            str(output_root),
            "--dry-run",
        ]
    )
    assert code == 0
    manifest = output_root / "manifests" / "candidates.jsonl"
    assert manifest.is_file()
    assert len(manifest.read_text(encoding="utf-8").splitlines()) == 1


def _write_ntu_skeleton(path: Path, frames=4):
    lines = [str(frames)]
    for frame_index in range(frames):
        lines.extend(["1", "1 0 0 0 0 0 0 0 0 2", "25"])
        for joint in range(25):
            x = frame_index * 0.1 + joint * 0.01
            y = joint * 0.02
            z = 2.0 + frame_index * 0.05
            lines.append(f"{x} {y} {z} 0 0 0 0 1 0 0 0 2")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ntu_skeleton_contact_sheets_are_rendered(tmp_path):
    skeleton = tmp_path / "S001C001P001R001A052.skeleton"
    _write_ntu_skeleton(skeleton)
    sheets, metadata = create_skeleton_contact_sheets(
        skeleton,
        tmp_path / "sheets",
        {
            "total_frames": 4,
            "uniform_frames": 2,
            "motion_frames": 2,
            "frames_per_sheet": 2,
            "cell_width": 192,
            "cell_height": 128,
            "jpeg_quality": 80,
        },
    )
    assert len(sheets) == 2
    assert all(path.is_file() for path in sheets)
    assert metadata["skeleton_format"] == "ntu25_3d"
    assert metadata["modality"] == "skeleton"


def test_candidate_quota_report_separates_modalities():
    records = []
    for label in (
        "normal_walk",
        "normal_run",
        "playful_chase",
        "playful_push",
        "conflict_chase",
        "conflict_push",
    ):
        records.extend(
            {"modality": "rgb", "candidate_labels": [label]} for _ in range(2)
        )
        records.extend(
            {"modality": "skeleton", "candidate_labels": [label]} for _ in range(3)
        )
    report = quota_report(
        records,
        {
            "schema_version": "test",
            "quotas": {"minimum_rgb_per_class": 2, "minimum_combined_per_class": 5},
        },
    )
    assert report["complete"]
    assert report["classes"]["playful_chase"] == {
        "rgb": 2,
        "skeleton": 3,
        "combined": 5,
        "rgb_shortfall": 0,
        "combined_shortfall": 0,
        "ready": True,
    }


def test_cross_modality_source_identity_prefers_skeleton():
    records = [
        {
            "path": "/rgb/shared.mp4",
            "modality": "rgb",
            "source_video_id": "youtube:shared",
        },
        {
            "path": "/skeleton/shared.json",
            "modality": "skeleton",
            "source_video_id": "youtube:shared",
        },
        {
            "path": "/rgb/unique.mp4",
            "modality": "rgb",
            "source_video_id": "youtube:unique",
        },
    ]
    kept, excluded = enforce_cross_modality_disjoint(records)
    assert {record["path"] for record in kept} == {
        "/skeleton/shared.json",
        "/rgb/unique.mp4",
    }
    assert [record["path"] for record in excluded] == ["/rgb/shared.mp4"]


def test_kinetics_rows_exclude_skeleton_source_video_ids(tmp_path):
    annotation = tmp_path / "train.csv"
    annotation.write_text(
        "label,youtube_id,time_start,time_end,split,is_cc\n"
        "wrestling,shared,0,10,train,0\n"
        "wrestling,rgb_only,0,10,train,0\n",
        encoding="utf-8",
    )
    groups = _kinetics_rows(annotation, ["wrestling"], {"shared"})
    assert [row["youtube_id"] for row in groups["wrestling"]] == ["rgb_only"]


def test_incremental_uav_human_and_hmdb51_sources(tmp_path):
    uav_root = tmp_path / "UAV-Human" / "RGBVideos"
    hmdb_root = tmp_path / "HMDB51" / "push"
    (uav_root / "A076").mkdir(parents=True)
    (uav_root / "A133").mkdir(parents=True)
    hmdb_root.mkdir(parents=True)
    (uav_root / "A076" / "push_001.avi").write_bytes(b"uav-push")
    (uav_root / "A133" / "chase_001.avi").write_bytes(b"uav-chase")
    (hmdb_root / "object_push.avi").write_bytes(b"hmdb-push")
    config = {
        "paths": {
            "uav_human_rgb_root": str(uav_root),
            "hmdb51_push_root": str(hmdb_root),
        },
        "uav_human_mappings": {
            "A076": ["playful_push", "conflict_push"],
            "A133": ["playful_chase", "conflict_chase"],
        },
        "hmdb51_push_mappings": {
            "push": ["playful_push", "conflict_push"],
        },
    }
    records = list(uav_human_rgb_records(config)) + list(hmdb51_push_records(config))
    assert len(records) == 3
    assert {record["source_dataset"] for record in records} == {"UAV-Human", "HMDB51"}
    assert {record["source_label"] for record in records} == {"A076", "A133", "push"}
    assert all(record["modality"] == "rgb" for record in records)


def test_dashscope_payload_controls_qwen_thinking(tmp_path, monkeypatch):
    image = tmp_path / "sheet.jpg"
    image.write_bytes(b"jpeg")
    monkeypatch.setenv("TEST_DASHSCOPE_API_KEY", "test-key")
    client = HostedVisionClient(
        {
            "provider": "dashscope",
            "model": "qwen3-vl-plus",
            "api_key_env": "TEST_DASHSCOPE_API_KEY",
            "thinking": True,
        },
        {"temperature": 0.0},
    )
    payload = client._payload([image], "audit")
    assert payload["enable_thinking"] is True


def test_labeled_rgb_and_skeleton_are_materialized_separately(tmp_path):
    config_path = (
        Path(__file__).resolve().parents[2]
        / "dataset_construction"
        / "configs"
        / "hosted_screening.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workflow = ScreeningWorkflow(config, tmp_path / "output")
    for modality, suffix in (("rgb", ".mp4"), ("skeleton", ".skeleton")):
        source = tmp_path / f"source_{modality}{suffix}"
        source.write_bytes(b"sample")
        workflow.materialize(
            {
                "sample": {
                    "sample_id": f"sample_{modality}",
                    "path": str(source),
                    "modality": modality,
                },
                "final": {"status": "accepted", "label": "normal_walk"},
            }
        )
        assert (
            tmp_path
            / "output"
            / "labeled"
            / modality
            / "normal_walk"
            / f"sample_{modality}{suffix}"
        ).exists()
