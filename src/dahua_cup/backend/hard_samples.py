"""Boolean hard-sample decision shared by the API and background workers."""

from __future__ import annotations

from typing import Any, Mapping, Optional


CONDITION_TITLES = (
    ("C1", "小模型预测不确定"),
    ("C2", "大小模型预测不同"),
    ("C3", "大小模型高置信度冲突"),
    ("C4", "时序预测不稳定"),
    ("C5", "类别稀有"),
)


def _state(value: Optional[bool]) -> str:
    if value is None:
        return "pending"
    return "yes" if value else "no"


def evaluate_hard_sample(
    prediction: Optional[Mapping[str, Any]],
    teacher: Optional[Mapping[str, Any]],
    *,
    margin_threshold: float,
    conflict_confidence_threshold: float,
    instability_threshold: float,
) -> dict:
    """Evaluate ``Hard(x)=C1 or C2 or C3 or C4 or C5`` without a score."""
    prediction = dict(prediction or {})
    teacher = dict(teacher or {})
    topk = list(prediction.get("topk") or [])
    gate = dict(prediction.get("teacher_gate") or {})
    teacher_result = dict(teacher.get("result") or {})
    teacher_ready = teacher.get("status") == "completed" and bool(teacher_result)

    margin = gate.get("top1_top2_margin")
    if margin is None and len(topk) >= 2:
        try:
            margin = float(topk[0]["score"]) - float(topk[1]["score"])
        except (KeyError, TypeError, ValueError):
            margin = None
    c1 = None if margin is None else float(margin) < margin_threshold

    c2: Optional[bool] = None
    c3: Optional[bool] = None
    if teacher_ready and topk:
        student_label = str(topk[0].get("label") or "")
        teacher_label = str(teacher_result.get("label") or "")
        c2 = bool(student_label and teacher_label and student_label != teacher_label)
        try:
            c3 = bool(
                c2
                and float(topk[0]["score"]) >= conflict_confidence_threshold
                and float(teacher_result["confidence"])
                >= conflict_confidence_threshold
            )
        except (KeyError, TypeError, ValueError):
            c3 = None

    instability = dict(gate.get("instability") or {})
    try:
        runs_compared = int(instability.get("runs_compared", 0))
    except (TypeError, ValueError):
        runs_compared = 0
    c4 = (
        None
        if runs_compared < 1
        else float(instability.get("score", 0.0)) >= instability_threshold
    )

    rare = prediction.get("rare_class")
    c5 = None if rare is None else bool(rare)
    values = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}
    conditions = [
        {"code": code, "title": title, "status": _state(values[code])}
        for code, title in CONDITION_TITLES
    ]
    matched = [code for code, _ in CONDITION_TITLES if values[code] is True]
    pending = [code for code, _ in CONDITION_TITLES if values[code] is None]
    is_hard = bool(matched)
    if is_hard:
        summary = "判定为难例：命中 {}。".format("、".join(matched))
    elif pending:
        summary = "当前未命中难例条件；{} 尚待补充证据。".format(
            "、".join(pending)
        )
    else:
        summary = "五项条件均未命中，判定为非难例。"
    return {
        "schema_version": "campus6_hard_decision.v1",
        "formula": "Hard(x) = C1 ∨ C2 ∨ C3 ∨ C4 ∨ C5",
        "is_hard": is_hard,
        "matched_conditions": matched,
        "matched_condition_count": len(matched),
        "pending_conditions": pending,
        "conditions": conditions,
        "summary": summary,
    }
