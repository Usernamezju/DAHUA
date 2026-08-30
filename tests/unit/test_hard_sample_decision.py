from dahua_cup.backend.hard_samples import evaluate_hard_sample


def evaluate(prediction, teacher=None):
    return evaluate_hard_sample(
        prediction,
        teacher,
        confidence_threshold=0.30,
        margin_threshold=0.15,
        conflict_confidence_threshold=0.70,
        instability_threshold=0.60,
    )


def prediction(*, confidence=0.25, margin=0.10, rare=False, instability=None):
    first = confidence
    value = {
        "topk": [
            {"label": "playful_push", "score": first},
            {"label": "conflict_push", "score": first - margin},
        ],
        "rare_class": rare,
        "teacher_gate": {},
    }
    if instability is not None:
        value["teacher_gate"]["instability"] = {
            "runs_compared": 2,
            "score": instability,
        }
    return value


def test_any_single_true_condition_makes_sample_hard():
    decision = evaluate(prediction(margin=0.20, rare=True))
    assert decision["is_hard"] is True
    assert decision["matched_conditions"] == ["C5"]
    assert decision["matched_condition_count"] == 1
    assert "hard_score" not in decision


def test_missing_teacher_and_temporal_evidence_are_pending_not_true():
    decision = evaluate(prediction(margin=0.20, rare=False))
    statuses = {item["code"]: item["status"] for item in decision["conditions"]}
    assert decision["is_hard"] is False
    assert statuses == {
        "C1": "no",
        "C2": "pending",
        "C3": "pending",
        "C4": "pending",
        "C5": "no",
    }


def test_teacher_disagreement_can_match_after_student_uncertainty_gate():
    value = prediction(confidence=0.28, margin=0.20, rare=False, instability=0.20)
    teacher = {
        "status": "completed",
        "result": {"label": "conflict_push", "confidence": 0.91},
    }
    decision = evaluate(value, teacher)
    assert decision["is_hard"] is True
    assert decision["matched_conditions"] == ["C2"]


def test_high_confidence_student_suppresses_every_condition_and_teacher_data():
    value = prediction(
        confidence=0.989, margin=0.01, rare=True, instability=0.95
    )
    teacher = {
        "status": "completed",
        "result": {"label": "normal_walk", "confidence": 0.91},
    }

    decision = evaluate(value, teacher)
    statuses = {item["code"]: item["status"] for item in decision["conditions"]}

    assert decision["is_hard"] is False
    assert decision["matched_conditions"] == []
    assert decision["pending_conditions"] == []
    assert decision["confidence_gate"] == {
        "threshold": 0.30,
        "student_confidence": 0.989,
        "eligible": False,
    }
    assert statuses == {code: "no" for code in ("C1", "C2", "C3", "C4", "C5")}
    assert "跳过难例判定与 Qwen" in decision["summary"]
