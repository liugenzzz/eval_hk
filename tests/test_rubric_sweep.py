import pandas as pd

from compare_rubrics import compare_runs, dz_pairwise_test


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
