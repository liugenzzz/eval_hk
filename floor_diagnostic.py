"""Bound how much of the V3 pass rate is the free-100 floor -- from output you already have.

No judge calls, no rerun. judge_detail_all.xlsx (or detail_<model>_vqa.xlsx) already stores
the five per-dimension scores, so the floor effect can be measured after the fact.

The one thing V3's output cannot tell you: whether a param_score/visual_score of exactly
100 means "every number was right" or "this question asked for no numbers, free 100".
So this reports a RANGE -- optimistic (treat every 100 as earned) and pessimistic (treat
every exact 100 on a nullable dimension as not-applicable and renormalize it away). The
truth is between them, and the gap itself tells you how much the rubric was guessing.

Usage:
    python floor_diagnostic.py /path/to/judge_detail_all.xlsx
    python floor_diagnostic.py /path/to/eval_report/   # scans detail_*_vqa.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

W100 = {"param_score": 0.35, "fact_score": 0.25, "visual_score": 0.15,
        "fabrication_score": 0.15, "style_score": 0.10}
NULLABLE = ("param_score", "visual_score")   # the two that award a free 100 in V3
PASS = 60.0
DIMS = list(W100)


def load(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(path.glob("detail_*_vqa.xlsx")) or sorted(path.glob("judge_detail_all.xlsx"))
        if not files:
            raise SystemExit(f"no detail_*_vqa.xlsx or judge_detail_all.xlsx under {path}")
        return pd.concat([pd.read_excel(f) for f in files], ignore_index=True)
    return pd.read_excel(path)


def quality_optimistic(row: pd.Series) -> float:
    """V3 as shipped: every dimension weighted, a 100 is a 100."""
    return sum(W100[d] * float(row[d]) for d in DIMS)


def quality_pessimistic(row: pd.Series) -> float:
    """V4-style: drop nullable dims that sit at exactly 100 (assume not-applicable),
    renormalize the remaining weights."""
    keep = {d: float(row[d]) for d in DIMS
            if not (d in NULLABLE and float(row[d]) == 100.0)}
    if not keep:
        return float("nan")
    w = sum(W100[d] for d in keep)
    return sum(W100[d] * v for d, v in keep.items()) / w


def report(df: pd.DataFrame) -> None:
    missing = [d for d in DIMS if d not in df.columns]
    if missing:
        raise SystemExit(f"columns not found: {missing}\n"
                         f"available: {list(df.columns)}\n"
                         f"(these only exist for rows scored with the 100-point rubric)")

    df = df.dropna(subset=DIMS).copy()
    if df.empty:
        raise SystemExit("no rows have all five dimension scores -- were these scored with V3?")

    df["q_opt"] = df.apply(quality_optimistic, axis=1)
    df["q_pes"] = df.apply(quality_pessimistic, axis=1)
    df["free_both"] = (df["param_score"] == 100) & (df["visual_score"] == 100)

    group_col = "model" if "model" in df.columns else None
    groups = df.groupby(group_col) if group_col else [("all", df)]

    for name, g in groups:
        n = len(g)
        print(f"\n{'='*72}\n{name}   n={n}\n{'='*72}")

        free = int(g["free_both"].sum())
        print(f"param==100 AND visual==100 (0.50 weight free)   {free:5d}  {free/n:6.1%}")
        for d in NULLABLE:
            c = int((g[d] == 100).sum())
            print(f"  {d} == 100 exactly                          {c:5d}  {c/n:6.1%}")

        pass_opt = float((g["q_opt"] >= PASS).mean())
        pass_pes = float((g["q_pes"] >= PASS).mean())
        print(f"\npass rate  optimistic (V3 as shipped)           {pass_opt:6.1%}")
        print(f"pass rate  pessimistic (free 100s removed)      {pass_pes:6.1%}")
        print(f"  -> up to {pass_opt - pass_pes:.1%} of the pass rate is floor, not quality")

        print(f"\nmean quality  optimistic {g['q_opt'].mean():6.2f}   "
              f"pessimistic {g['q_pes'].mean():6.2f}")

        # Passes that only cleared the line because of the free weight.
        floor_pass = g[(g["q_opt"] >= PASS) & (g["q_pes"] < PASS)]
        print(f"\nrows that pass ONLY with the free 100s:         {len(floor_pass):5d}  "
              f"{len(floor_pass)/n:6.1%}")
        if len(floor_pass):
            print("  their fact_score distribution (low = genuinely wrong answers passing):")
            print("   ", floor_pass["fact_score"].describe()[["mean", "25%", "50%", "75%"]]
                  .round(1).to_dict())
            bad = int((floor_pass["fact_score"] <= 40).sum())
            print(f"    fact_score <= 40 (misidentified / conflicting):  {bad}  "
                  f"({bad/len(floor_pass):.1%} of floor passes)")

        # Resolution actually used.
        print(f"\nquality_score spread   optimistic std {g['q_opt'].std():5.2f}   "
              f"pessimistic std {g['q_pes'].std():5.2f}")
        print(f"distinct hit values (what the report averages):  "
              f"{g['q_opt'].ge(PASS).nunique()}  <- this is why the headline moves so little")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    report(load(Path(sys.argv[1])))
