from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd


def bootstrap_binary_ci(values: Sequence[object], n_bootstrap: int = 1000, seed: int = 42) -> tuple[float, float]:
    arr = pd.Series(values).dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return math.nan, math.nan
    if len(arr) == 1:
        value = float(arr[0])
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return round(float(low), 4), round(float(high), 4)


def summarize_binary_metric(
    data: pd.DataFrame,
    group_cols: list[str],
    metric_col: str = "hit",
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if data.empty:
        return pd.DataFrame(columns=[*group_cols, "score", "n", "ci_low", "ci_high"])
    for group_key, group in data.groupby(group_cols, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        values = group[metric_col].dropna().astype(float)
        low, high = bootstrap_binary_ci(values, n_bootstrap=bootstrap_n, seed=seed)
        row = {col: value for col, value in zip(group_cols, group_key)}
        row.update(
            {
                "score": round(float(values.mean()), 4) if len(values) else math.nan,
                "n": int(len(values)),
                "ci_low": low,
                "ci_high": high,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_metric_summary(
    details: pd.DataFrame,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    scopes = [
        ("overall", ["model", "dataset"]),
        ("capability", ["model", "dataset", "category"]),
        ("content_type", ["model", "dataset", "l2-category"]),
    ]
    for scope, cols in scopes:
        if all(col in details.columns for col in cols):
            summary = summarize_binary_metric(details, cols, "hit", bootstrap_n, seed)
            summary.insert(0, "scope", scope)
            parts.append(summary)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def make_wide_summary(long_summary: pd.DataFrame) -> pd.DataFrame:
    if long_summary.empty:
        return pd.DataFrame()
    rows: dict[str, dict[str, object]] = {}
    for _, row in long_summary.iterrows():
        model = str(row["model"])
        dataset = str(row["dataset"])
        scope = str(row["scope"])
        if scope == "overall":
            metric = f"{dataset}:overall"
        elif scope == "capability":
            metric = f"{dataset}:cap:{row.get('category')}"
        elif scope == "content_type":
            metric = f"{dataset}:ct:{row.get('l2-category')}"
        else:
            continue
        rows.setdefault(model, {"model": model})
        rows[model][metric] = row["score"]
        rows[model][f"{metric}:ci_low"] = row["ci_low"]
        rows[model][f"{metric}:ci_high"] = row["ci_high"]
        rows[model][f"{metric}:n"] = row["n"]
    return pd.DataFrame(rows.values())


def make_cross_tables(details: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    required = {"model", "dataset", "category", "l2-category", "hit"}
    if not required.issubset(details.columns):
        return tables
    for model, model_df in details.groupby("model"):
        rows: list[pd.DataFrame] = []
        for dataset, ds_df in model_df.groupby("dataset"):
            mean = ds_df.pivot_table(index="category", columns="l2-category", values="hit", aggfunc="mean")
            count = ds_df.pivot_table(index="category", columns="l2-category", values="hit", aggfunc="count")
            mean = mean.add_suffix("_score")
            count = count.add_suffix("_n")
            table = pd.concat([mean, count], axis=1).reset_index()
            table.insert(0, "dataset", dataset)
            rows.append(table)
        if rows:
            tables[str(model)] = pd.concat(rows, ignore_index=True)
    return tables


MIN_CATEGORY_N = 30
"""Categories with fewer rows than this are excluded from total_score.

Without a gate, category_weights.get(cat, 1.0) gives every category equal weight
regardless of size, so a category holding 2 rows and one holding 200 each contribute half
the headline number -- each of those 2 rows carries 100x the weight of a single-turn row,
and a 2/2 result (an ordinary coin flip landing twice) pins 50% of the total at 1.0.
The category is still reported on its own row; it just stops driving the total."""


def make_weighted_score_summary(
    details: pd.DataFrame,
    category_weights: dict[str, float],
    min_category_n: int = MIN_CATEGORY_N,
) -> pd.DataFrame:
    """One row per model: weighted total score plus each category's raw score and n.

    Emits three totals so a size-imbalanced split cannot hide behind one number:
      total_score        -- category-weighted over categories with >= min_category_n rows
      total_score_by_n   -- sample-weighted over every row, ignoring category weights
      total_score_ungated-- the old behaviour, every category weighted equally
    When the first two disagree materially, the split is imbalanced and the category
    weights are doing more work than the data supports.
    """
    if details.empty or not {"model", "category", "hit"}.issubset(details.columns):
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for model, group in details.groupby("model"):
        cat_scores: dict[str, float] = {}
        cat_n: dict[str, int] = {}
        for category, cat_group in group.groupby("category"):
            values = cat_group["hit"].dropna().astype(float)
            if len(values):
                cat_scores[str(category)] = round(float(values.mean()), 4)
                cat_n[str(category)] = int(len(values))

        counted = {c: s for c, s in cat_scores.items() if cat_n[c] >= min_category_n}
        dropped = sorted(c for c in cat_scores if c not in counted)

        def _weighted(scores: dict[str, float]) -> float:
            w_sum = sum(category_weights.get(c, 1.0) for c in scores)
            if not w_sum:
                return math.nan
            return sum(category_weights.get(c, 1.0) * s for c, s in scores.items()) / w_sum

        total_n = sum(cat_n.values())
        row: dict[str, object] = {
            "model": str(model),
            "total_score": round(_weighted(counted), 4) if counted else math.nan,
            "total_score_by_n": (round(sum(cat_scores[c] * cat_n[c] for c in cat_scores) / total_n, 4)
                                 if total_n else math.nan),
            "total_score_ungated": round(_weighted(cat_scores), 4) if cat_scores else math.nan,
            "n_total": int(total_n),
            "categories_in_total": ",".join(sorted(counted)),
            "categories_excluded_small_n": ",".join(f"{c}(n={cat_n[c]})" for c in dropped),
        }
        for category in sorted(cat_scores):
            row[category] = cat_scores[category]
            row[f"{category}_n"] = cat_n[category]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_pairwise_vs_baseline(pairwise: pd.DataFrame, bootstrap_n: int = 1000, seed: int = 42) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_sets = [
        ("overall", ["model", "baseline_model"]),
        ("capability", ["model", "baseline_model", "category"]),
        ("content_type", ["model", "baseline_model", "l2-category"]),
    ]
    for scope, cols in group_sets:
        if not set(cols).issubset(pairwise.columns):
            continue
        for group_key, group in pairwise.groupby(cols, dropna=False):
            if not isinstance(group_key, tuple):
                group_key = (group_key,)
            row = {"scope": scope, **{col: value for col, value in zip(cols, group_key)}}
            n = len(group)
            row["n"] = int(n)
            row["win_rate"] = round(float((group["outcome_for_model"] == "win").mean()), 4) if n else math.nan
            row["tie_rate"] = round(float((group["outcome_for_model"] == "tie").mean()), 4) if n else math.nan
            row["loss_rate"] = round(float((group["outcome_for_model"] == "loss").mean()), 4) if n else math.nan
            low, high = bootstrap_binary_ci((group["outcome_for_model"] == "win").astype(int), bootstrap_n, seed)
            row["win_ci_low"] = low
            row["win_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)
