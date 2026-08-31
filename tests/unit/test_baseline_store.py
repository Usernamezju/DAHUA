from dahua_cup.backend.store import ReviewStore


def _record(label="normal_walk"):
    return {
        "sample_id": "campus6_0000_test",
        "video_path": "/server/annotations.pkl",
        "source_dataset": "Campus6_initial",
        "source_label": label,
        "manual_label": label,
        "reviewer": "official_initial_annotation",
        "reason_code": "official_ground_truth",
        "note": "split=train",
    }


def test_baseline_import_is_idempotent_and_preserves_later_review(tmp_path):
    store = ReviewStore(tmp_path / "review.sqlite3")
    assert store.import_baseline_records([_record()]) == {
        "inserted": 1,
        "existing": 0,
    }
    store.submit_review(
        "campus6_0000_test",
        "reviewer-a",
        "normal_run",
        "manual_correction",
    )

    assert store.import_baseline_records([_record()]) == {
        "inserted": 0,
        "existing": 1,
    }
    sample = store.get_sample("campus6_0000_test")
    assert sample["manual_label"] == "normal_run"
    assert sample["reviewer"] == "reviewer-a"
    # 普通审核接受/修改标签不会自动进入难例池。
    assert store.count_incremental_samples(None) == 0
    assert store.count_incremental_samples("9999-01-01T00:00:00+00:00") == 0


def test_restart_marks_incomplete_jobs_failed(tmp_path):
    store = ReviewStore(tmp_path / "review.sqlite3")
    store.import_baseline_records([_record()])
    job = store.create_job("campus6_0000_test", "teacher")
    store.update_job(job["job_id"], status="running")
    assert store.recover_incomplete_jobs() == 1
    recovered = store.get_job(job["job_id"])
    assert recovered["status"] == "failed"
    assert recovered["finished_at"]


def test_sample_summary_omits_video_and_pseudo_record_payload(tmp_path):
    store = ReviewStore(tmp_path / "review.sqlite3")
    store.import_baseline_records([_record()])

    result = store.list_samples(summary=True)

    assert result["total"] == 1
    summary = result["items"][0]
    assert summary["sample_id"] == "campus6_0000_test"
    assert summary["source_dataset"] == "Campus6_initial"
    assert "video_path" not in summary
    assert "pseudo_record" not in summary
    assert summary["workflow_status"] == "complete"


def test_human_label_disagreement_enters_reversible_incremental_hard_pool(tmp_path):
    store = ReviewStore(tmp_path / "review.sqlite3")
    store.import_baseline_records([_record()])
    store.submit_review(
        "campus6_0000_test",
        "reviewer-a",
        "normal_run",
        "manual_correction",
        student_snapshot={
            "status": "completed",
            "top6": [{"label": "normal_walk", "score": 0.90}],
        },
    )

    sample = store.get_sample("campus6_0000_test")
    assert sample["incremental_pool"] == 1
    assert sample["incremental_reason"] == "human_label_disagrees_with_student"
    assert any(
        event["event_type"] == "incremental_hard_sample_enqueued"
        for event in store.recent_events()
    )

    store.undo_last_review("campus6_0000_test", "reviewer-a")
    restored = store.get_sample("campus6_0000_test")
    assert restored["incremental_pool"] == 0
    assert restored["incremental_reason"] == ""


def test_human_labelled_hard_sample_enters_pool_even_when_student_agrees(tmp_path):
    store = ReviewStore(tmp_path / "review.sqlite3")
    store.import_baseline_records([_record()])
    store.submit_review(
        "campus6_0000_test",
        "reviewer-a",
        "normal_walk",
        "human_reviewed_hard_sample",
        student_snapshot={
            "status": "completed",
            "top6": [{"label": "normal_walk", "score": 0.90}],
        },
    )

    sample = store.get_sample("campus6_0000_test")
    assert sample["incremental_pool"] == 1
    assert sample["incremental_reason"] == "human_reviewed_hard_sample"
    assert store.count_incremental_samples(None) == 1
