"""Parsers for all four pointwise rubric versions, on one common output scale.

Every version returns ``quality_score`` on 0-100 and ``hit`` as 0/1. That matters for the
comparison: with a shared scale the four versions can go into the same report and into
compare_rubrics.py without per-version special-casing, and the paired effect size d_z is
directly comparable across them.

Dispatch reads the ``rubric`` tag the prompts now emit. When it is absent -- old rows in a
cache or a detail xlsx scored before the tag existed -- it falls back to key-presence
heuristics and parses with the ORIGINAL semantics of that format, so historical output
keeps reading back the way it was scored.

Shared design across V2/V3/V4: a dimension that does not apply to the question returns
``null``, is dropped, and the remaining weights are renormalized. The versions this
replaces awarded a free top score instead, which handed away 0.50 of the total weight in
both V2 and V3 -- in V2 that put a totally wrong answer at 0.60 out of 1.00.
"""

from __future__ import annotations

import json
import re
from typing import Any

PASS_THRESHOLD = 60.0

# ---- V2: coarse 1-5 Likert, three axes -------------------------------------------
V2_WEIGHTS = {"param_score": 0.50, "fact_score": 0.35, "style_score": 0.15}
V2_NULLABLE = ("param_score",)

# ---- V3: 0-100, five axes, pure weighted mean ------------------------------------
V3_WEIGHTS = {"param_score": 0.35, "fact_score": 0.25, "visual_score": 0.15,
              "fabrication_score": 0.15, "relevance_score": 0.10}
V3_NULLABLE = ("param_score", "fact_score", "visual_score")

# ---- V4: accuracy axes + fabrication multiplier + equipment gate -----------------
V4_ACC_WEIGHTS = {"fact_score": 0.45, "param_score": 0.30, "visual_score": 0.25}
V4_NULLABLE = ("fact_score", "param_score", "visual_score")
V4_FABRICATION_FLOOR = 0.50   # fabrication 0 -> accuracy halved
V4_WRONG_EQUIPMENT_CAP = 30.0
V4_STYLE_BLEND = 0.0          # style deliberately out of the headline number

# ---- legacy formats, kept so old rows still read back correctly -------------------
LEGACY_100_WEIGHTS = {"param_score": 0.35, "fact_score": 0.25, "visual_score": 0.15,
                      "fabrication_score": 0.15, "style_score": 0.10}
LEGACY_5_WEIGHTS = {"param_score": 0.50, "fact_score": 0.35, "style_score": 0.15}

ALL_COLUMNS = (
    "quality_score", "accuracy_score", "equipment_correct", "applicable_dims",
    "failure_type", "param_score", "fact_score", "visual_score",
    "fabrication_score", "style_score", "relevance_score",
)


# ==================================================================================
# JSON extraction
# ==================================================================================

def _iter_json_objects(text: str):
    depth, start, in_str, escape = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                yield text[start:i + 1]


def extract_json_object(content: str) -> dict[str, Any]:
    """Pick the verdict object out of the judge's raw reply.

    A greedy r'\\{.*\\}' spans two objects when the judge restates the format example
    before answering; a naive first-brace scan grabs the example instead of the verdict.
    Either way the row ends up a parse failure. So: parse every complete object and
    prefer the last one that carries a verdict key.
    """
    text = str(content or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    candidates = []
    for span in _iter_json_objects(text):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
    if not candidates:
        return json.loads(text)   # surface the decode error with the raw content
    verdict_keys = {"rubric", "equipment_correct", "correct", "relevance_score",
                    "fabrication_score", "fact_score"}
    for obj in reversed(candidates):
        if verdict_keys & obj.keys():
            return obj
    return candidates[-1]


# ==================================================================================
# helpers
# ==================================================================================

_NULLISH = {"null", "none", "n/a", "na", "不适用", ""}


def _score(value: object, field: str, lo: float, hi: float, nullable: bool) -> float | None:
    if value is None or (isinstance(value, str) and value.strip().lower() in _NULLISH):
        if not nullable:
            raise ValueError(f"{field} must be scored, got null")
        return None
    score = float(value)
    if not (lo <= score <= hi):
        raise ValueError(f"{field}={value!r} out of range [{lo:g},{hi:g}]")
    return score


def _weighted_renormalized(scores: dict[str, float], weights: dict[str, float]) -> float:
    if not scores:
        raise ValueError("no applicable dimension was scored")
    total_w = sum(weights[k] for k in scores)
    return sum(weights[k] * v for k, v in scores.items()) / total_w


def _blank() -> dict[str, object]:
    return {col: None for col in ALL_COLUMNS}


def _collect(obj: dict[str, Any], weights: dict[str, float], nullable: tuple[str, ...],
             lo: float, hi: float) -> tuple[dict[str, float], dict[str, float | None]]:
    applicable: dict[str, float] = {}
    parsed: dict[str, float | None] = {}
    for field in weights:
        if field not in obj:
            raise ValueError(f"missing {field} (must be a number or explicit null)")
        value = _score(obj[field], field, lo, hi, nullable=field in nullable)
        parsed[field] = value
        if value is not None:
            applicable[field] = value
    return applicable, parsed


# ==================================================================================
# per-version parsers
# ==================================================================================

def parse_v1(obj: dict[str, Any]) -> dict[str, object]:
    if "correct" not in obj:
        raise ValueError("missing correct")
    correct = obj["correct"] is True or str(obj["correct"]).strip().lower() == "true"
    eq_raw = obj.get("equipment_correct")
    equipment_correct = None
    if eq_raw is not None:
        equipment_correct = eq_raw is True or str(eq_raw).strip().lower() == "true"
    failure = obj.get("failure_type")
    if isinstance(failure, str) and failure.strip().lower() in _NULLISH:
        failure = None
    out = _blank()
    out.update({
        "quality_score": 100.0 if correct else 0.0,
        "equipment_correct": equipment_correct,
        "failure_type": failure,
    })
    return {"hit": 1 if correct else 0,
            "reason": str(obj.get("reason", obj.get("reasoning", ""))), **out}


def parse_v2(obj: dict[str, Any]) -> dict[str, object]:
    applicable, parsed = _collect(obj, V2_WEIGHTS, V2_NULLABLE, 1, 5)
    # (x-1)/4 maps 1..5 onto 0..100. The retired version divided by 5, which left a dead
    # 20% at the bottom of the axis -- an all-1s answer scored 0.2 instead of 0.
    scaled = {k: (v - 1.0) * 25.0 for k, v in applicable.items()}
    quality = _weighted_renormalized(scaled, V2_WEIGHTS)
    out = _blank()
    out.update({
        "quality_score": round(quality, 2),
        "applicable_dims": ",".join(sorted(applicable)),
        **parsed,
    })
    return {"hit": 1 if quality >= PASS_THRESHOLD else 0,
            "reason": str(obj.get("reasoning", obj.get("reason", ""))), **out}


def parse_v3(obj: dict[str, Any]) -> dict[str, object]:
    applicable, parsed = _collect(obj, V3_WEIGHTS, V3_NULLABLE, 0, 100)
    quality = _weighted_renormalized(applicable, V3_WEIGHTS)
    out = _blank()
    out.update({
        "quality_score": round(quality, 2),
        "applicable_dims": ",".join(sorted(applicable)),
        **parsed,
    })
    return {"hit": 1 if quality >= PASS_THRESHOLD else 0,
            "reason": str(obj.get("reasoning", obj.get("reason", ""))), **out}


def parse_v4(obj: dict[str, Any]) -> dict[str, object]:
    if "equipment_correct" not in obj:
        raise ValueError("missing equipment_correct")
    equipment_correct = (obj["equipment_correct"] is True
                         or str(obj["equipment_correct"]).strip().lower() == "true")

    applicable, parsed = _collect(obj, V4_ACC_WEIGHTS, V4_NULLABLE, 0, 100)
    accuracy = _weighted_renormalized(applicable, V4_ACC_WEIGHTS)

    fabrication = _score(obj.get("fabrication_score"), "fabrication_score", 0, 100, nullable=False)
    style = _score(obj.get("style_score"), "style_score", 0, 100, nullable=False)
    assert fabrication is not None and style is not None

    quality = accuracy * (V4_FABRICATION_FLOOR
                          + (1.0 - V4_FABRICATION_FLOOR) * (fabrication / 100.0))
    if V4_STYLE_BLEND:
        quality = (1.0 - V4_STYLE_BLEND) * quality + V4_STYLE_BLEND * style
    if not equipment_correct:
        quality = min(quality, V4_WRONG_EQUIPMENT_CAP)

    out = _blank()
    out.update({
        "quality_score": round(quality, 2),
        "accuracy_score": round(accuracy, 2),
        "equipment_correct": equipment_correct,
        "applicable_dims": ",".join(sorted(applicable)),
        "fabrication_score": fabrication,
        "style_score": style,
        **{k: v for k, v in parsed.items()},
    })
    return {"hit": 1 if quality >= PASS_THRESHOLD else 0,
            "reason": str(obj.get("reasoning", obj.get("reason", ""))), **out}


def _parse_binary(obj: dict[str, Any], want_equipment: bool) -> dict[str, object]:
    """v3b / v4b: native binary verdict with aspect-level failure attribution.

    quality_score is 100/0 so these sit on the same axis as every other version and go
    into compare_rubrics without special-casing. aspect_failures lands in failure_type as
    a comma-joined string, reusing the column v1 already writes.
    """
    if "correct" not in obj:
        raise ValueError("missing correct")
    correct = obj["correct"] is True or str(obj["correct"]).strip().lower() == "true"

    equipment_correct = None
    eq_raw = obj.get("equipment_correct")
    if eq_raw is not None:
        equipment_correct = eq_raw is True or str(eq_raw).strip().lower() == "true"
    elif want_equipment:
        raise ValueError("missing equipment_correct")

    failures = obj.get("aspect_failures", [])
    if isinstance(failures, str):
        failures = [t.strip() for t in re.split(r"[,\s]+", failures) if t.strip()]
    if not isinstance(failures, list):
        failures = []
    failures = [str(f).strip().lower() for f in failures if str(f).strip().lower() not in _NULLISH]

    # A false verdict with no attribution is still usable, but flag it so the rate is
    # visible -- if it is high the judge is not following the attribution instruction and
    # the failure clustering will be empty.
    if not correct and not failures:
        failures = ["unattributed"]

    out = _blank()
    out.update({
        "quality_score": 100.0 if correct else 0.0,
        "equipment_correct": equipment_correct,
        "failure_type": ",".join(failures) if failures else None,
    })
    return {"hit": 1 if correct else 0,
            "reason": str(obj.get("reason", obj.get("reasoning", ""))), **out}


def parse_v3b(obj: dict[str, Any]) -> dict[str, object]:
    return _parse_binary(obj, want_equipment=False)


def parse_v4b(obj: dict[str, Any]) -> dict[str, object]:
    return _parse_binary(obj, want_equipment=True)


# ---- legacy: original semantics, for reading rows scored before the rewrite --------

def parse_legacy_100(obj: dict[str, Any]) -> dict[str, object]:
    scores = {k: _score(obj[k], k, 0, 100, nullable=False) for k in LEGACY_100_WEIGHTS}
    quality = sum(LEGACY_100_WEIGHTS[k] * v for k, v in scores.items())  # no renormalization
    out = _blank()
    out.update({"quality_score": round(quality, 2), **scores})
    return {"hit": 1 if quality >= PASS_THRESHOLD else 0,
            "reason": str(obj.get("reasoning", obj.get("reason", ""))), **out}


def parse_legacy_5(obj: dict[str, Any]) -> dict[str, object]:
    scores = {k: _score(obj[k], k, 1, 5, nullable=False) for k in LEGACY_5_WEIGHTS}
    weighted = sum(LEGACY_5_WEIGHTS[k] * v for k, v in scores.items())
    out = _blank()
    out.update({"quality_score": round(weighted / 5.0 * 100, 2), **scores})
    return {"hit": round(weighted / 5.0, 4),   # original returned the fractional score
            "reason": str(obj.get("reasoning", obj.get("reason", ""))), **out}


# ==================================================================================
# dispatch
# ==================================================================================

PARSERS = {"v1": parse_v1, "v2": parse_v2, "v3": parse_v3, "v3b": parse_v3b,
           "v4": parse_v4, "v4b": parse_v4b}


def parse_pointwise(content: str) -> dict[str, object]:
    obj = extract_json_object(content)

    tag = str(obj.get("rubric", "")).strip().lower()
    if tag in PARSERS:
        return PARSERS[tag](obj)

    # Untagged: infer from keys, using each format's original semantics.
    if "equipment_correct" in obj and "fabrication_score" in obj:
        return parse_v4(obj)
    if "relevance_score" in obj:
        return parse_v3(obj)
    if "visual_score" in obj or "fabrication_score" in obj:
        return parse_legacy_100(obj)
    if "correct" in obj:
        # v3b/v4b carry aspect_failures; v1 carries a single failure_type
        if "aspect_failures" in obj:
            return _parse_binary(obj, want_equipment=False)
        return parse_v1(obj)
    if {"param_score", "fact_score", "style_score"} & obj.keys():
        return parse_legacy_5(obj)
    raise ValueError(f"unrecognized judge output, keys={sorted(obj.keys())}")


if __name__ == "__main__":
    samples = {
        "v1 pass": '{"rubric":"v1","correct":true,"equipment_correct":true,"failure_type":null,"reason":"ok"}',
        "v1 fail": '{"rubric":"v1","correct":false,"equipment_correct":false,"failure_type":"equipment","reason":"认错"}',
        "v2 worst": '{"rubric":"v2","param_score":1,"fact_score":1,"style_score":1}',
        "v2 free-param worst": '{"rubric":"v2","param_score":null,"fact_score":1,"style_score":1}',
        "v2 best": '{"rubric":"v2","param_score":5,"fact_score":5,"style_score":5}',
        "v3 wrong equip, no num/look": '{"rubric":"v3","param_score":null,"fact_score":0,"visual_score":null,"fabrication_score":60,"relevance_score":60}',
        "v3 good": '{"rubric":"v3","param_score":100,"fact_score":100,"visual_score":100,"fabrication_score":100,"relevance_score":80}',
        "v4 wrong equip": '{"rubric":"v4","equipment_correct":false,"fact_score":20,"param_score":null,"visual_score":null,"fabrication_score":40,"style_score":30}',
        "v4 good": '{"rubric":"v4","equipment_correct":true,"fact_score":100,"param_score":null,"visual_score":null,"fabrication_score":100,"style_score":80}',
        "legacy 100 (old free-100 hole)": '{"param_score":100,"fact_score":0,"visual_score":100,"fabrication_score":60,"style_score":60}',
        "v3b pass": '{"rubric":"v3b","correct":true,"aspect_failures":[]}',
        "v3b fail": '{"rubric":"v3b","correct":false,"aspect_failures":["fact","fabrication"]}',
        "v4b gate fail": '{"rubric":"v4b","correct":false,"equipment_correct":false,"aspect_failures":["fact"]}',
        "v4b pass": '{"rubric":"v4b","correct":true,"equipment_correct":true,"aspect_failures":[]}',
        "v4b unattributed": '{"rubric":"v4b","correct":false,"equipment_correct":true,"aspect_failures":[]}',
        "legacy 1-5": '{"param_score":5,"fact_score":3,"style_score":1}',
    }
    print(f"{'sample':34s} {'quality':>8s} {'hit':>7s}  dims")
    print("-" * 88)
    for name, payload in samples.items():
        r = parse_pointwise(payload)
        print(f"{name:34s} {str(r['quality_score']):>8s} {str(r['hit']):>7s}  "
              f"{r['applicable_dims'] or r['failure_type'] or '-'}")
