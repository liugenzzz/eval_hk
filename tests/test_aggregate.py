import pandas as pd

from eval_tool.aggregate import bootstrap_binary_ci, summarize_binary_metric


def test_bootstrap_binary_ci_is_reproducible_and_bounds_mean():
    values = [1, 1, 0, 1, 0, 1]

    low, high = bootstrap_binary_ci(values, n_bootstrap=200, seed=123)

    assert 0.0 <= low <= high <= 1.0
    assert low <= sum(values) / len(values) <= high


def test_summarize_binary_metric_groups_by_capability_and_counts_samples():
    data = pd.DataFrame(
        [
            {"model": "base", "dataset": "mcq", "hit": 1, "category": "P1"},
            {"model": "base", "dataset": "mcq", "hit": 0, "category": "P1"},
            {"model": "base", "dataset": "mcq", "hit": 1, "category": "R1"},
        ]
    )

    summary = summarize_binary_metric(
        data,
        group_cols=["model", "dataset", "category"],
        metric_col="hit",
        bootstrap_n=100,
        seed=7,
    )

    p1 = summary[summary["category"] == "P1"].iloc[0]
    assert p1["n"] == 2
    assert p1["score"] == 0.5
    assert {"ci_low", "ci_high"}.issubset(summary.columns)
