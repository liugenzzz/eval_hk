from __future__ import annotations

import math
import warnings

import pandas as pd


def pointwise_length_control(
    data: pd.DataFrame,
    baseline_model: str,
    min_rows: int = 10,
) -> pd.DataFrame:
    required = {"model", "hit", "pred_len"}
    models = sorted(str(m) for m in data.get("model", pd.Series(dtype=str)).dropna().unique())
    if not required.issubset(data.columns) or len(data) < min_rows or len(models) < 2:
        return _raw_pointwise(data, "insufficient_data")
    if data["pred_len"].nunique(dropna=True) <= 1:
        return _raw_pointwise(data, "pred_len_no_variance")
    try:
        import statsmodels.api as sm
    except Exception:
        return _raw_pointwise(data, "statsmodels_unavailable")

    frame = data[["model", "hit", "pred_len"]].dropna().copy()
    frame["model"] = frame["model"].astype(str)
    frame["hit"] = frame["hit"].astype(int)
    frame["pred_len"] = frame["pred_len"].astype(float)
    dummies = pd.get_dummies(frame["model"], prefix="model", dtype=float)
    baseline_col = f"model_{baseline_model}"
    if baseline_col in dummies.columns:
        dummies = dummies.drop(columns=[baseline_col])
    x = pd.concat([frame[["pred_len"]], dummies], axis=1)
    x = sm.add_constant(x, has_constant="add")
    y = frame["hit"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(y, x).fit(disp=False)
    except Exception as exc:
        return _raw_pointwise(data, f"logit_failed:{type(exc).__name__}")

    ref_len = float(frame["pred_len"].mean())
    rows: list[dict[str, object]] = []
    for model in models:
        row = {col: 0.0 for col in x.columns}
        row["const"] = 1.0
        row["pred_len"] = ref_len
        col = f"model_{model}"
        if col in row:
            row[col] = 1.0
        pred_x = pd.DataFrame([row], columns=x.columns)
        score = float(fit.predict(pred_x)[0])
        rows.append(
            {
                "model": model,
                "length_control_score": round(score, 4),
                "reference_pred_len": round(ref_len, 2),
                "length_control_warning": "",
            }
        )
    return pd.DataFrame(rows)


def _raw_pointwise(data: pd.DataFrame, warning: str) -> pd.DataFrame:
    if data.empty or "model" not in data.columns:
        return pd.DataFrame(columns=["model", "length_control_score", "reference_pred_len", "length_control_warning"])
    ref_len = float(data["pred_len"].mean()) if "pred_len" in data.columns and len(data) else math.nan
    rows = []
    for model, group in data.groupby("model"):
        rows.append(
            {
                "model": model,
                "length_control_score": round(float(group["hit"].mean()), 4) if "hit" in group else math.nan,
                "reference_pred_len": round(ref_len, 2) if not math.isnan(ref_len) else math.nan,
                "length_control_warning": warning,
            }
        )
    return pd.DataFrame(rows)


def pairwise_length_control(pairwise: pd.DataFrame, min_rows: int = 10) -> pd.DataFrame:
    if pairwise.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for model, group in pairwise.groupby("model"):
        non_tie = group[group["outcome_for_model"].isin(["win", "loss"])].copy()
        raw = float((group["outcome_for_model"] == "win").mean()) if len(group) else math.nan
        if len(non_tie) < min_rows or "len_diff" not in non_tie.columns or non_tie["len_diff"].nunique() <= 1:
            rows.append({"model": model, "pairwise_lc_win_rate": round(raw, 4), "pairwise_lc_warning": "insufficient_data"})
            continue
        try:
            import statsmodels.api as sm

            x = sm.add_constant(non_tie[["len_diff"]].astype(float), has_constant="add")
            y = (non_tie["outcome_for_model"] == "win").astype(int)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.Logit(y, x).fit(disp=False)
            pred_x = pd.DataFrame([{"const": 1.0, "len_diff": 0.0}], columns=x.columns)
            score = float(fit.predict(pred_x)[0])
            rows.append({"model": model, "pairwise_lc_win_rate": round(score, 4), "pairwise_lc_warning": ""})
        except Exception as exc:
            rows.append({"model": model, "pairwise_lc_win_rate": round(raw, 4), "pairwise_lc_warning": f"logit_failed:{type(exc).__name__}"})
    return pd.DataFrame(rows)
