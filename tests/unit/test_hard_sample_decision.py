from dahua_cup.backend.hard_samples import evaluate_hard_sample


def evaluate(prediction, teacher=None):
    return evaluate_hard_sample(
        prediction,
        teacher,
        margin_threshold=0.15,
        conflict_confidence_threshold=0.70,
        instability_threshold=0.60,
    )


def prediction(*, margin=0.10, rare=False, instability=None):
    first = 0.50
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
    decision = evaluate(prediction(margin=0.30, rare=True))
    assert decision["is_hard"] is True
    assert decision["matched_conditions"] == ["C5"]
    assert decision["matched_condition_count"] == 1
    assert "hard_score" not in decision


def test_missing_teacher_and_temporal_evidence_are_pending_not_true():
    decision = evaluate(prediction(margin=0.30, rare=False))
    statuses = {item["code"]: item["status"] for item in decision["conditions"]}
    assert decision["is_hard"] is False
    assert statuses == {
        "C1": "no",
        "C2": "pending",
        "C3": "pending",
        "C4": "pending",
        "C5": "no",
    }


def test_disagreement_and_high_confidence_conflict_can_both_match():
    value = prediction(margin=0.20, rare=False, instability=0.20)
    value["topk"][0]["score"] = 0.82
    teacher = {
        "status": "completed",
        "result": {"label": "conflict_push", "confidence": 0.91},
    }
    decision = evaluate(value, teacher)
    assert decision["is_hard"] is True
    assert decision["matched_conditions"] == ["C2", "C3"]
    assert decision["formula"] == "Hard(x) = C1 ∨ C2 ∨ C3 ∨ C4 ∨ C5"
