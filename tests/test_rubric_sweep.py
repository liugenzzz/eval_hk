import json

import pandas as pd
import pytest

from compare_rubrics import compare_runs, dz_pairwise_test
from eval_tool.config import load_pipeline_config
from eval_tool.pipeline import PipelineError, run_sweep
from eval_tool.rubrics import RubricError


def _scored_frame(models=("base", "sft1", "sft2"), count=24, quality=False):
    rows = []
    offsets = {"base": 0.0, "sft1": 0.15, "sft2": -0.08}
    for index in range(count):
        for model in models:
            raw = 0.45 + offsets[model] + ((index % 5) - 2) * 0.03
            row = {
                "index": str(index),
                "model": model,
                "hit": float(raw >= 0.5),
                "human_score": raw * 10,
            }
            if quality:
                row["quality_score"] = raw
            rows.append(row)
    return pd.DataFrame(rows)


def test_compare_runs_returns_each_rubric_and_challenger():
    runs = {
        "v3": _scored_frame(),
        "v4": _scored_frame(quality=True),
    }

    result = compare_runs(runs, baseline="base", n_bootstrap=50)

    assert set(zip(result["rubric"], result["model"])) == {
        ("v3", "sft1"),
        ("v3", "sft2"),
        ("v4", "sft1"),
        ("v4", "sft2"),
    }
    assert dict(zip(result["rubric"], result["metric"])) == {
        "v3": "hit",
        "v4": "quality_score",
    }
    assert "spearman_vs_human" in result.columns


def test_compare_runs_keeps_error_row_when_only_baseline_is_present():
    result = compare_runs(
        {"v4": _scored_frame(models=("base",))},
        baseline="base",
        n_bootstrap=20,
    )

    assert result.loc[0, "rubric"] == "v4"
    assert result.loc[0, "error"] == "only the baseline model is present"


def test_dz_pairwise_test_groups_each_challenger_independently():
    runs = {
        "v3": _scored_frame(quality=True),
        "v4": _scored_frame(quality=True),
    }

    result = dz_pairwise_test(
        runs,
        baseline="base",
        metric="quality_score",
        n_bootstrap=50,
    )

    assert set(result["model"]) == {"sft1", "sft2"}
    assert result.groupby("model")["pair"].nunique().to_dict() == {
        "sft1": 1,
        "sft2": 1,
    }


def _pipeline_config(tmp_path, *, scored=False):
    base = {"name": "base", "model_path": "models/base"}
    if scored:
        base["scored"] = {"vqa": "scored/base.xlsx"}
    raw = {
        "tsv_dir": "tsv",
        "datasets": {"vqa": "aero_vqa"},
        "work_dir": "work",
        "out_dir": "reports/eval_report",
        "cache_dir": "cache",
        "models": [
            base,
            {"name": "sft1", "model_path": "models/sft1"},
            {"name": "sft2", "model_path": "models/sft2"},
        ],
        "baseline_model": "base",
        "infer": {},
    }
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return load_pipeline_config(path)


def test_sweep_runs_serially_and_writes_comparison(tmp_path, monkeypatch):
    config = _pipeline_config(tmp_path)
    calls = []

    def fake_eval(eval_config):
        calls.append(eval_config.out_dir.name)
        detail = eval_config.out_dir / "judge_detail_all.xlsx"
        detail.parent.mkdir(parents=True, exist_ok=True)
        _scored_frame().to_excel(detail, index=False)
        return {"judge_detail_all.xlsx": detail}

    monkeypatch.setattr("eval_tool.pipeline.run_eval", fake_eval)

    result = run_sweep(config, ["v3", "v4"], n_bootstrap=50)

    assert calls == ["eval_report_v3", "eval_report_v4"]
    assert result.comparison_path.is_file()
    comparison = pd.read_csv(result.comparison_path)
    assert comparison["rubric"].tolist() == ["v3", "v3", "v4", "v4"]


def test_sweep_rejects_scored_overrides(tmp_path):
    with pytest.raises(PipelineError, match=r"sweep cannot use models\[\]\.scored"):
        run_sweep(_pipeline_config(tmp_path, scored=True), ["v3", "v4"])


@pytest.mark.parametrize(
    ("rubrics", "message"),
    [([], "at least one rubric"), (["v3", "v3"], "duplicate rubrics")],
)
def test_sweep_rejects_empty_or_duplicate_rubrics(tmp_path, rubrics, message):
    with pytest.raises(RubricError, match=message):
        run_sweep(_pipeline_config(tmp_path), rubrics)


def test_sweep_stops_after_first_failed_rubric(tmp_path, monkeypatch):
    config = _pipeline_config(tmp_path)
    calls = []

    def fail(eval_config):
        calls.append(eval_config.out_dir.name)
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr("eval_tool.pipeline.run_eval", fail)

    with pytest.raises(RuntimeError, match="judge unavailable"):
        run_sweep(config, ["v3", "v4"])
    assert calls == ["eval_report_v3"]


def test_sweep_requires_detail_artifact(tmp_path, monkeypatch):
    config = _pipeline_config(tmp_path)
    monkeypatch.setattr("eval_tool.pipeline.run_eval", lambda eval_config: {})

    with pytest.raises(PipelineError, match="v4 did not produce judge_detail_all"):
        run_sweep(config, ["v4"])


def test_sweep_propagates_model_filter_and_no_pairwise(tmp_path, monkeypatch):
    config = _pipeline_config(tmp_path)
    seen = []

    def fake_eval(eval_config):
        seen.append(
            ([model.name for model in eval_config.models], eval_config.do_pairwise)
        )
        detail = eval_config.out_dir / "judge_detail_all.xlsx"
        detail.parent.mkdir(parents=True, exist_ok=True)
        _scored_frame(models=("base", "sft2")).to_excel(detail, index=False)
        return {"judge_detail_all.xlsx": detail}

    monkeypatch.setattr("eval_tool.pipeline.run_eval", fake_eval)

    run_sweep(
        config,
        ["v4"],
        model_names=["base", "sft2"],
        no_pairwise=True,
        n_bootstrap=20,
    )

    assert seen == [(["base", "sft2"], False)]
