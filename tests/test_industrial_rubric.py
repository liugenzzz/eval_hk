"""Numerical and schema regression tests for the independent industrial rubric."""

from pathlib import Path

import pytest

from eval_tool.industrial_rubric import INDUSTRIAL_COLUMNS, INDUSTRIAL_PROMPT_MARKER, parse_industrial_v1


def verdict(**updates):
    data = dict(rubric="industrial_v1", reference_status="usable", object_score=100,
                defect_score=80, appearance_score=60, location_score=40,
                fabrication_score=100, style_score=100, critical_errors=[], reasoning="有证据支持。")
    return {**data, **updates}


def test_prompt_declares_dispatch_marker():
    prompt = Path(__file__).resolve().parents[1] / "prompts" / "judge_industrial_pointwise_v1.txt"
    assert prompt.read_text(encoding="utf-8").splitlines()[0] == INDUSTRIAL_PROMPT_MARKER


def test_weighted_accuracy_and_multiplier():
    result = parse_industrial_v1(verdict(fabrication_score=60))
    assert result["accuracy_score"] == 74
    assert result["quality_score"] == 59.2
    assert result["hit"] == 0
    assert set(INDUSTRIAL_COLUMNS) <= result.keys()


def test_optional_attributes_renormalize_and_style_does_not_change_score():
    result = parse_industrial_v1(verdict(object_score=None, defect_score=100,
                                        appearance_score=None, location_score=None, style_score=0))
    assert result["quality_score"] == 100
    assert result["applicable_dims"] == "defect_score"
    assert result["hit"] == 1


@pytest.mark.parametrize("error", ["wrong_object", "missed_defect", "false_alarm", "wrong_defect_type"])
def test_critical_error_caps_quality(error):
    result = parse_industrial_v1(verdict(critical_errors=[error]))
    assert result["quality_score"] == 30
    assert result["accuracy_score"] == 74
    assert result["hit"] == 0
    assert result["industrial_critical_errors"] == error


@pytest.mark.parametrize("status", ["conflict", "insufficient"])
def test_unusable_evidence_is_not_model_failure(status):
    result = parse_industrial_v1(verdict(reference_status=status,
        **{key: None for key in ("object_score", "defect_score", "appearance_score",
                                 "location_score", "fabrication_score", "style_score")}))
    assert result["quality_score"] is result["accuracy_score"] is result["hit"] is None
    assert result["industrial_reference_status"] == status
    assert result["reason"]


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), 0.5, 100.0, -1, 101, "80", [], {}])
def test_invalid_scores_rejected(value):
    with pytest.raises(ValueError):
        parse_industrial_v1(verdict(defect_score=value))


@pytest.mark.parametrize("updates", [
    {"rubric": "v4"}, {"reference_status": "unknown"}, {"reference_status": []},
    {"reasoning": " "}, {"reasoning": 1}, {"critical_errors": "false_alarm"},
    {"critical_errors": ["unknown"]}, {"critical_errors": [{}]},
    {"critical_errors": ["false_alarm", "false_alarm"]},
    {"fabrication_score": None}, {"style_score": None},
    {"object_score": None, "defect_score": None, "appearance_score": None, "location_score": None},
    {"reference_status": "conflict"}, {"reference_status": "insufficient"},
])
def test_invalid_schema_rejected(updates):
    with pytest.raises(ValueError):
        parse_industrial_v1(verdict(**updates))


def test_missing_nullable_key_is_rejected():
    data = verdict()
    del data["location_score"]
    with pytest.raises(ValueError, match="missing"):
        parse_industrial_v1(data)


def test_pass_threshold_and_fabrication_floor():
    assert parse_industrial_v1(verdict(object_score=None, defect_score=60,
        appearance_score=None, location_score=None))["hit"] == 1
    assert parse_industrial_v1(verdict(object_score=None, defect_score=100,
        appearance_score=None, location_score=None, fabrication_score=0))["quality_score"] == 50


def test_unusable_reference_cannot_contain_critical_errors():
    data = verdict(reference_status="conflict", critical_errors=["false_alarm"])
    for key in ("object_score", "defect_score", "appearance_score", "location_score", "fabrication_score", "style_score"):
        data[key] = None
    with pytest.raises(ValueError):
        parse_industrial_v1(data)
