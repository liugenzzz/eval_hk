"""Industrial-only, offline summaries of existing pointwise scores.

This does not run a judge or infer defect-attribute accuracy from a prose score.
Run after eval: python -m eval_tool.industrial_report --details ... --out ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .breakdown import DimSpec, make_breakdown


METRICS = (
    "quality_score", "hit", "object_score", "defect_score", "appearance_score", "location_score",
    "fabrication_score", "style_score",
)
ATTRIBUTES = (
    "object_category", "source_domain", "defect_type", "defect_form",
    "defect_location", "defect_status", "defect_bucket",
)


def summarize_industrial(
    details: pd.DataFrame,
    *,
    datasets: tuple[str, ...] = ("industrial_zh", "industrial_en"),
    min_n: int = 30,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Keep language datasets separate and exclude null scores per metric."""
    if min_n < 1 or bootstrap_n < 1:
        raise ValueError("min_n and bootstrap_n must be positive")
    if not {"model", "dataset"}.issubset(details.columns):
        raise ValueError("details must contain model and dataset columns")
    frame = details[details["dataset"].isin(datasets)].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("no industrial dataset rows found")
    if "quality_score" not in frame:
        raise ValueError("details must contain pointwise quality_score")
    # The shared judge_detail_all export keeps category but omits meta.*.
    # Industrial profiles explicitly map category to object_category.
    if "meta.object_category" not in frame and "category" in frame:
        frame["meta.object_category"] = frame["category"]
    frame["industrial.all"] = "all"
    dims = [DimSpec("overall", "industrial.all", min_n=min_n)]
    dims.extend(DimSpec(key, f"meta.{key}", min_n=min_n) for key in ATTRIBUTES)
    result = make_breakdown(
        frame, dims, metrics=METRICS, bootstrap_n=bootstrap_n, seed=seed,
    )
    if result.empty:
        return pd.DataFrame(columns=["model", "dataset", "dim", "value", "metric", "score",
                                     "n", "ci_low", "ci_high", "status"])
    return result


def summarize_industrial_status(
    details: pd.DataFrame, *, datasets: tuple[str, ...] = ("industrial_zh", "industrial_en"),
) -> pd.DataFrame:
    """Show excluded rows alongside scored rows, including all-unscorable groups."""
    frame = details[details["dataset"].isin(datasets)].copy()
    if frame.empty:
        return pd.DataFrame()
    if "meta.object_category" not in frame and "category" in frame:
        frame["meta.object_category"] = frame["category"]
    rows = []
    for (model, dataset), group in frame.groupby(["model", "dataset"], dropna=False):
        groups = [("overall", "all", group)]
        if "meta.object_category" in group:
            groups.extend(("object_category", str(value), part) for value, part in
                          group.groupby("meta.object_category", dropna=False))
        for dim, value, part in groups:
            status = part.get("industrial_reference_status", pd.Series("", index=part.index)).fillna("")
            reason = part.get("judge_reason", pd.Series("", index=part.index)).fillna("").astype(str)
            scored = pd.to_numeric(part["quality_score"], errors="coerce").notna()
            conflict = ~scored & status.eq("conflict")
            insufficient = ~scored & status.eq("insufficient")
            failed = ~scored & ~conflict & ~insufficient & reason.str.startswith("[judge_error]")
            missing = ~scored & ~conflict & ~insufficient & reason.str.startswith("[missing_prediction]")
            rows.append({
                "model": model, "dataset": dataset, "dim": dim, "value": value,
                "n_total": len(part), "n_scored": int(scored.sum()),
                "n_reference_conflict": int(conflict.sum()),
                "n_reference_insufficient": int(insufficient.sum()),
                "n_judge_error": int(failed.sum()), "n_missing_prediction": int(missing.sum()),
                "n_other_unscored": int((~(scored | conflict | insufficient | failed | missing)).sum()),
            })
    return pd.DataFrame(rows)


def summarize_industrial_errors(
    details: pd.DataFrame, *, datasets: tuple[str, ...] = ("industrial_zh", "industrial_en"),
) -> pd.DataFrame:
    """Report judge attributions on scored, usable references with explicit denominators."""
    frame = details[details["dataset"].isin(datasets)].copy()
    if frame.empty:
        return pd.DataFrame()
    for name in ("quality_score", "industrial_reference_status"):
        if name not in frame:
            raise ValueError(f"details must contain {name}")
    frame = frame[pd.to_numeric(frame["quality_score"], errors="coerce").notna()
                  & frame["industrial_reference_status"].eq("usable")].copy()
    if frame.empty:
        return pd.DataFrame()
    if "meta.defect_status" not in frame:
        return pd.DataFrame(columns=["model", "dataset", "dim", "value", "flag",
                                     "n_eligible", "n_flagged", "rate"])
    if frame["meta.defect_status"].isna().any():
        raise ValueError("scored industrial details have incomplete meta.defect_status")
    if "industrial_critical_errors" not in frame:
        raise ValueError("details must contain industrial_critical_errors for scored rows")
    flags = ("missed_defect", "false_alarm", "wrong_object", "wrong_defect_type")
    rows = []
    groups = [("overall", "all", frame)]
    for field in ("defect_status", "defect_bucket", "object_category", "source_domain"):
        column = f"meta.{field}"
        if column in frame:
            groups.extend((field, str(value), group) for value, group in
                          frame.groupby(column, dropna=False))
    for dim, value, group in groups:
        for (model, dataset), part in group.groupby(["model", "dataset"], dropna=False):
            status = part.get("meta.defect_status", pd.Series("", index=part.index))
            parsed = part["industrial_critical_errors"].fillna("").astype(str).map(
                lambda text: set(text.split(",")) if text else set()
            )
            for flag in flags:
                eligible = status.eq("abnormal") if flag in {"missed_defect", "wrong_defect_type"} else (
                    status.eq("normal") if flag == "false_alarm" else pd.Series(True, index=part.index)
                )
                denominator = int(eligible.sum())
                n_flagged = int(parsed[eligible].map(lambda found: flag in found).sum())
                rows.append({"model": model, "dataset": dataset, "dim": dim, "value": value,
                             "flag": flag, "n_eligible": denominator, "n_flagged": n_flagged,
                             "rate": n_flagged / denominator if denominator else None})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["industrial_zh", "industrial_en"])
    parser.add_argument("--min-n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    details = pd.concat([
        pd.read_excel(path) if path.suffix.lower() == ".xlsx" else pd.read_csv(path)
        for path in args.details
    ], ignore_index=True)
    if {"model", "dataset", "index"}.issubset(details.columns):
        if details.duplicated(["model", "dataset", "index"]).any():
            raise ValueError("duplicate model/dataset/index rows; do not combine overlapping exports")
    report = summarize_industrial(
        details, datasets=tuple(args.datasets), min_n=args.min_n, seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8-sig", newline="") as handle:
        report.to_csv(handle, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
