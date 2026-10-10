"""Validated annotations and separate metrics for abnormal-event video clips.

The released NWPU/Avenue masks locate events but do not name them. Only
human-verified records may be scored as abnormal-event recognition.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


TAXONOMY_VERSION = "park_abnormal_v1"
TIME_AGREEMENT_TOLERANCE_S = 0.5
# A deliberately small, reviewable closed set; extend only with annotated clips.
TYPE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "running": ("running", "run", "奔跑", "跑步"),
    "throwing_object": ("throwing an object", "throwing objects", "throw object", "抛物", "扔东西", "投掷物品"),
    "fighting": ("fighting", "physical fight", "打架", "斗殴", "肢体冲突"),
    "falling": ("falling", "fall down", "摔倒", "跌倒"),
    "wrong_way": ("wrong-way movement", "moving against traffic", "逆行", "反向行走"),
    "loitering": ("loitering", "徘徊", "长时间逗留"),
    "climbing": ("climbing", "攀爬", "翻越"),
    "trespassing": ("trespassing", "unauthorized entry", "闯入", "非法进入"),
}


def _require_nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _interval(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain start_s and end_s")
    start, end = value.get("start_s"), value.get("end_s")
    if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        raise ValueError(f"{name} times must be numeric")
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
        raise ValueError(f"{name} requires 0 <= start_s < end_s and finite times")
    return float(start), float(end)


def validate_annotation(record: Mapping[str, Any], *, require_verified: bool = True) -> dict[str, Any]:
    """Validate one frozen reference; never infer a class from a video mask."""
    if record.get("taxonomy_version") != TAXONOMY_VERSION:
        raise ValueError(f"taxonomy_version must be {TAXONOMY_VERSION}")
    clip_id = _require_nonempty(record.get("clip_id"), "clip_id")
    video = _require_nonempty(record.get("video"), "video")
    if Path(video).suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise ValueError("video must reference a video file")
    event_type = _require_nonempty(record.get("event_type"), "event_type")
    if event_type not in TYPE_SYNONYMS:
        raise ValueError(f"unknown event_type: {event_type}")
    description = _require_nonempty(record.get("description"), "description")
    start, end = _interval(record.get("interval"), "interval")
    duration = record.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < end:
        raise ValueError("duration_s must contain the event interval")
    reviews = record.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise ValueError("reviews must contain exactly two independent entries")
    reviewers = []
    review_intervals = []
    for index, review in enumerate(reviews):
        if not isinstance(review, Mapping):
            raise ValueError(f"reviews[{index}] must be an object")
        reviewers.append(_require_nonempty(review.get("reviewer"), f"reviews[{index}].reviewer"))
        if review.get("event_type") not in TYPE_SYNONYMS:
            raise ValueError(f"reviews[{index}].event_type is not in the taxonomy")
        review_intervals.append(_interval(review.get("interval"), f"reviews[{index}].interval"))
        _require_nonempty(review.get("description"), f"reviews[{index}].description")
    if len(set(reviewers)) != 2:
        raise ValueError("reviewers must be distinct")
    agreement = record.get("review_agreement")
    if not isinstance(agreement, Mapping) or set(agreement) != {"event_type", "interval", "description"} or any(type(value) is not bool for value in agreement.values()):
        raise ValueError("review_agreement must contain three boolean decisions")
    actual_type_agreement = reviews[0]["event_type"] == reviews[1]["event_type"] == event_type
    actual_interval_agreement = all(
        abs(first - second) <= TIME_AGREEMENT_TOLERANCE_S
        for first, second in zip(review_intervals[0], review_intervals[1])
    ) and all(
        abs(final - review) <= TIME_AGREEMENT_TOLERANCE_S
        for final, review in zip((start, end), review_intervals[0])
    )
    if agreement["event_type"] != actual_type_agreement:
        raise ValueError("event_type agreement conflicts with independent reviews or final label")
    if agreement["interval"] != actual_interval_agreement:
        raise ValueError("interval agreement conflicts with independent reviews or final interval")
    status = record.get("verification")
    if status not in {"draft", "disputed", "verified"}:
        raise ValueError("verification must be draft, disputed, or verified")
    if require_verified and status != "verified":
        raise ValueError(f"clip {clip_id} has not been verified")
    if status == "verified":
        if not all(agreement.values()):
            adjudicator = _require_nonempty(record.get("adjudicator"), "adjudicator")
            if adjudicator in reviewers:
                raise ValueError("adjudicator must differ from both reviewers")
            _require_nonempty(record.get("adjudication_reason"), "adjudication_reason")
        elif record.get("adjudicator"):
            if record["adjudicator"] in reviewers:
                raise ValueError("adjudicator must differ from both reviewers")
    return {**record, "interval": {"start_s": start, "end_s": end}, "duration_s": float(duration)}


def load_annotations(path: str | Path, *, require_verified: bool = True) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = validate_annotation(json.loads(line), require_verified=require_verified)
                clip_id = record["clip_id"]
                if clip_id in records:
                    raise ValueError(f"duplicate clip_id: {clip_id}")
                records[clip_id] = record
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: no annotations")
    return records


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase = phrase.casefold()
    if re.search(r"[a-z]", phrase):
        pattern = r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])"
    else:
        pattern = re.escape(phrase)
    for match in re.finditer(pattern, text):
        prefix = text[max(0, match.start() - 12):match.start()]
        if not re.search(r"(?:没有|并非|不是|未|无|not\s+|no\s+)$", prefix):
            return True
    return False


def match_event_types(answer: str) -> set[str]:
    """Map answer text to every mentioned closed-set type; ambiguous answers stay ambiguous."""
    text = answer.casefold()
    return {key for key, terms in TYPE_SYNONYMS.items() if any(_contains_phrase(text, term) for term in terms)}


def temporal_iou(reference: Mapping[str, Any], predicted: Mapping[str, Any]) -> float:
    a0, a1 = _interval(reference, "reference interval")
    b0, b1 = _interval(predicted, "predicted interval")
    intersection = max(0.0, min(a1, b1) - max(a0, b0))
    return intersection / (max(a1, b1) - min(a0, b0))


DescriptionJudge = Callable[[str, str], float]


def score_predictions(
    references: Mapping[str, Mapping[str, Any]],
    predictions: Iterable[Mapping[str, Any]],
    *,
    require_time: bool = False,
    description_judge: DescriptionJudge | None = None,
) -> list[dict[str, Any]]:
    """Score separate type/time/semantic dimensions on exactly matched clip IDs.

    description_judge may wrap the existing video_ref_v1 evaluator. No semantic
    score is fabricated when it is absent.
    """
    predicted_by_id: dict[str, Mapping[str, Any]] = {}
    for prediction in predictions:
        clip_id = _require_nonempty(prediction.get("clip_id"), "prediction clip_id")
        if clip_id in predicted_by_id:
            raise ValueError(f"duplicate prediction clip_id: {clip_id}")
        predicted_by_id[clip_id] = prediction
    if set(references) != set(predicted_by_id):
        raise ValueError(f"clip IDs differ: missing={sorted(set(references) - set(predicted_by_id))}, extra={sorted(set(predicted_by_id) - set(references))}")
    rows = []
    for clip_id, reference in references.items():
        validate_annotation(reference)
        prediction = predicted_by_id[clip_id]
        answer = _require_nonempty(prediction.get("answer"), "answer")
        matched = match_event_types(answer)
        event_type = prediction.get("event_type")
        if event_type is not None:
            if event_type not in TYPE_SYNONYMS:
                raise ValueError(f"unknown predicted event_type: {event_type}")
        score = None
        if description_judge is not None:
            score = description_judge(reference["description"], answer)
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
                raise ValueError("description judge must return a score in [0, 1]")
        interval = prediction.get("interval")
        if require_time and interval is None:
            raise ValueError(f"clip {clip_id}: predicted interval required")
        rows.append({
            "clip_id": clip_id,
            "reference_type": reference["event_type"],
            "matched_types": sorted(matched),
            "type_hit": matched == {reference["event_type"]},
            "structured_type_hit": event_type == reference["event_type"] if event_type is not None else None,
            "temporal_iou": temporal_iou(reference["interval"], interval) if interval is not None else None,
            "description_score": score,
        })
    return rows
