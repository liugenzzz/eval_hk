"""Domain mismatches fail before scoring or expensive inference."""

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from eval_tool import pipeline, scorers
from eval_tool.config import load_pipeline_config
from eval_tool.judge import JudgeClient, JudgeSettings
from eval_tool.rubrics import RubricError, apply_rubric, validate_industrial_rubric
from eval_tool.scorers.industrial import validate_scored_industrial


ROOT = Path(__file__).resolve().parents[1]


def industrial():
    return load_pipeline_config(ROOT / "industrial.example.json")


def test_industrial_cannot_be_switched_to_equipment_rubric():
    config = industrial().to_eval_config()
    with pytest.raises(RubricError, match="incompatible"):
        apply_rubric(config, "v4")
    applied = apply_rubric(config, "industrial_v1")
    assert applied.judge.fingerprint == config.judge.fingerprint
    assert applied.out_dir.name.endswith("_industrial_v1")
    assert not scorers.get("industrial_describe").pairwise


def test_book_cannot_use_industrial_rubric_and_legacy_prompt_is_unchanged():
    config = load_pipeline_config(ROOT / "book.example.json").to_eval_config()
    original = config.judge.fingerprint
    with pytest.raises(RubricError, match="separate"):
        apply_rubric(config, "industrial_v1")
    assert config.judge.fingerprint == original
    assert apply_rubric(config, "v4").judge.fingerprint == original


def test_wrong_config_prompt_is_detected_without_cli_override():
    config = industrial().to_eval_config()
    with pytest.raises(RubricError, match="incompatible"):
        validate_industrial_rubric(replace(config, judge=JudgeSettings()))


def test_all_and_sweep_validate_before_any_model_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must reject mismatched rubric before work")

    monkeypatch.setattr(pipeline, "run_inference", forbidden)
    monkeypatch.setattr(pipeline, "run_eval", forbidden)
    with pytest.raises(RubricError):
        pipeline.run_all(industrial(), rubric="v4")
    with pytest.raises(RubricError):
        pipeline.run_sweep(industrial(), ["industrial_v1", "v4"])


def test_wrong_judge_reply_cannot_silently_switch_scoring_rules(monkeypatch):
    monkeypatch.setattr(JudgeClient, "_post", lambda *a, **kw: json.dumps({"rubric": "v4"}))
    with pytest.raises(ValueError, match="both use rubric"):
        JudgeClient(industrial().judge).judge_pointwise("question", "reference", "prediction")
    monkeypatch.setattr(JudgeClient, "_post", lambda *a, **kw: json.dumps({"rubric": "industrial_v1"}))
    with pytest.raises(ValueError, match="both use rubric"):
        JudgeClient(JudgeSettings()).judge_pointwise("question", "reference", "prediction")


def test_equipment_scored_file_is_not_reused_for_industrial():
    with pytest.raises(ValueError, match="rerun from predictions"):
        validate_scored_industrial(pd.DataFrame([{"quality_score": 99, "fact_score": 100}]))


def test_disabled_pointwise_keeps_an_explicit_unscored_result():
    ctx = scorers.ScoringContext(
        dataset_key="industrial_zh", kind="industrial_describe", model_name="base",
        judge_client=JudgeClient(industrial().judge), do_pointwise=False,
    )
    result = scorers.get("industrial_describe").score(
        pd.DataFrame([{"answer": "reference", "prediction": "candidate"}]), ctx)
    assert result.quality_score.isna().all()
    assert result.hit.isna().all()
