"""Independent industrial inspection rubric; no equipment/book scoring changes.

Weights and gates are initial engineering choices, not empirically calibrated.
"""

from __future__ import annotations

from typing import Any

RUBRIC_TAG = "industrial_v1"
INDUSTRIAL_PROMPT_MARKER = "评分版本：industrial_v1"
WEIGHTS = {"object_score": 0.20, "defect_score": 0.45,
           "appearance_score": 0.20, "location_score": 0.15}
PASS_THRESHOLD = 60.0
CRITICAL_ERRORS = frozenset({"wrong_object", "missed_defect", "false_alarm", "wrong_defect_type"})
INDUSTRIAL_COLUMNS = (*WEIGHTS, "industrial_reference_status", "industrial_critical_errors", "rubric")
_SCORES = (*WEIGHTS, "fabrication_score", "style_score")


def parse_industrial_v1(obj: dict[str, Any]) -> dict[str, Any]:
    """Validate a decoded judge object, returning report columns on a 0–100 scale."""
    if not isinstance(obj, dict) or obj.get("rubric") != RUBRIC_TAG:
        raise ValueError("industrial verdict must declare rubric='industrial_v1'")
    status = obj.get("reference_status")
    if not isinstance(status, str) or status not in {"usable", "conflict", "insufficient"}:
        raise ValueError("invalid industrial reference_status")
    reason = obj.get("reasoning")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("industrial reasoning must be a non-empty string")
    errors = obj.get("critical_errors")
    if not isinstance(errors, list) or any(not isinstance(e, str) or e not in CRITICAL_ERRORS for e in errors):
        raise ValueError("invalid industrial critical_errors")
    if len(set(errors)) != len(errors):
        raise ValueError("duplicate industrial critical_errors")
    scores = {}
    for name in _SCORES:
        if name not in obj:
            raise ValueError(f"missing industrial score: {name}")
        value = obj[name]
        if value is not None and (type(value) is not int or not 0 <= value <= 100):
            raise ValueError(f"{name} must be an integer in 0..100 or null")
        scores[name] = value
    applicable = [name for name in WEIGHTS if scores[name] is not None]
    accuracy = quality = hit = None
    if status != "usable":
        if any(value is not None for value in scores.values()) or errors:
            raise ValueError("unusable reference requires all scores null and no critical_errors")
    else:
        if scores["fabrication_score"] is None or scores["style_score"] is None:
            raise ValueError("usable reference requires fabrication_score and style_score")
        if not applicable:
            raise ValueError("usable reference requires at least one applicable accuracy dimension")
        accuracy = sum(WEIGHTS[name] * scores[name] for name in applicable) / sum(WEIGHTS[name] for name in applicable)
        quality = accuracy * (0.5 + 0.5 * scores["fabrication_score"] / 100)
        if errors:
            quality = min(quality, 30.0)
        hit = int(quality >= PASS_THRESHOLD)
        accuracy, quality = round(accuracy, 4), round(quality, 4)
    return {**scores, "quality_score": quality, "accuracy_score": accuracy,
            "hit": hit, "reason": reason.strip(), "applicable_dims": ",".join(sorted(applicable)),
            "industrial_reference_status": status, "industrial_critical_errors": ",".join(errors),
            "rubric": RUBRIC_TAG}
