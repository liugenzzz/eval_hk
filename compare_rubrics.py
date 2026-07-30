"""Compare rubric versions without touching the report pipeline.

Reads one judge_detail_all.xlsx per rubric version and answers the two questions that
actually decide which rubric to keep:

  1. Does it separate the models?  -> paired effect size d_z = mean(diff) / sd(diff),
     which is invariant under affine rescaling and therefore comparable across a binary
     rubric, a [0.2, 1.0] rubric and a [0, 100] rubric.
  2. Does it agree with humans?    -> Spearman for continuous rubrics, kappa for binary
     ones. Only runs if the file has a `human_score` column.

A rubric can win on (2) and lose badly on (1). That combination is exactly what a muted
delta looks like: the judge is right on average and still cannot tell the models apart.

Usage:
    python compare_rubrics.py --baseline base \\
        v1=runs/v1/judge_detail_all.xlsx \\
        v2=runs/v2/judge_detail_all.xlsx \\
        v4=runs/v4/judge_detail_all.xlsx

    # pick which column is the score for a given version (default: auto)
    python compare_rubrics.py --baseline base --metric hit v1=... v2=...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def pick_metric(df: pd.DataFrame, requested: str | None) -> str:
    """quality_score when the rubric produces one, else hit."""
    if requested:
        return requested
    if "quality_score" in df.columns and df["quality_score"].notna().any():
        return "quality_score"
    return "hit"


def paired_stats(df: pd.DataFrame, baseline: str, metric: str,
                 n_bootstrap: int = 5000, seed: int = 42) -> dict[str, object]:
    wide = df.pivot_table(index="index", columns="model", values=metric, aggfunc="first")
    if baseline not in wide.columns:
        return {"error": f"baseline '{baseline}' not in {list(wide.columns)}"}
    others = [c for c in wide.columns if c != baseline]
    if not others:
        return {"error": "only the baseline model is present"}
    target = others[0]

    pair = wide[[target, baseline]].dropna()
    if len(pair) < 10:
        return {"error": f"only {len(pair)} paired rows"}

    a = pair[target].to_numpy(dtype=float)
    b = pair[baseline].to_numpy(dtype=float)
    d = a - b

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_bootstrap, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])

    sd = d.std(ddof=1)
    return {
        "model": target,
        "n_paired": len(d),
        f"{baseline}_mean": round(float(b.mean()), 4),
        f"{target}_mean": round(float(a.mean()), 4),
        "delta": round(float(d.mean()), 4),
        "delta_ci": f"[{lo:+.4f}, {hi:+.4f}]",
        "significant": bool(lo > 0 or hi < 0),
        # scale-invariant -- this is the number to compare ACROSS rubric versions
        "effect_size_dz": round(float(d.mean() / sd), 3) if sd > 0 else float("nan"),
        "better/equal/worse": f"{int((d>0).sum())}/{int((d==0).sum())}/{int((d<0).sum())}",
        "distinct_values": int(pd.unique(np.concatenate([a, b])).size),
    }


def human_agreement(df: pd.DataFrame, metric: str) -> dict[str, object]:
    if "human_score" not in df.columns:
        return {}
    sub = df[[metric, "human_score"]].dropna()
    if len(sub) < 10:
        return {"agreement_n": len(sub)}
    j = sub[metric].to_numpy(dtype=float)
    h = sub["human_score"].to_numpy(dtype=float)

    # Spearman without scipy: Pearson on ranks
    def rank(x):
        return pd.Series(x).rank().to_numpy()
    jr, hr = rank(j), rank(h)
    spearman = float(np.corrcoef(jr, hr)[0, 1])

    # kappa: binarize each side. A rubric that is already 0/1 is used as-is -- taking its
    # median would put the cut at 0 and collapse it to all-ones, which reports kappa=0
    # for a judge that may agree with humans perfectly.
    j_is_binary = set(np.unique(j)) <= {0.0, 1.0}
    jb = j.astype(int) if j_is_binary else (j >= np.median(j)).astype(int)
    hb = (h >= np.median(h)).astype(int)
    po = float((jb == hb).mean())
    pe = jb.mean() * hb.mean() + (1 - jb.mean()) * (1 - hb.mean())
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")

    # how much of the human-measured gap this rubric recovers, in sd units
    return {
        "agreement_n": len(sub),
        "spearman_vs_human": round(spearman, 4),
        "kappa_at_median": round(float(kappa), 4),
        "judge_sd/human_sd": round(float(j.std() / (h.std() or 1.0)), 3),
    }


def dz_pairwise_test(runs: dict[str, pd.DataFrame], baseline: str, metric: str | None,
                     n_bootstrap: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Are two rubrics' effect sizes actually distinguishable on this sample?

    d_z has a standard error around sqrt((1 + d^2/2)/n), which at n~200 is roughly 0.09 --
    so a gap of 0.07 between two rubrics means nothing on its own. But the rubrics scored
    the SAME items, so their d_z values are strongly correlated and the difference has a
    much tighter interval than either d_z does. This resamples the shared index set once
    per iteration, recomputes every rubric's d_z on that resample, and takes pairwise
    differences -- which uses that correlation instead of throwing it away.

    A CI that straddles 0 means the two rubrics are indistinguishable here and the choice
    has to be made on agreement with human labels instead.
    """
    diffs: dict[str, pd.Series] = {}
    for label, df in runs.items():
        col = pick_metric(df, metric)
        wide = df.pivot_table(index="index", columns="model", values=col, aggfunc="first")
        if baseline not in wide.columns:
            continue
        others = [c for c in wide.columns if c != baseline]
        if not others:
            continue
        pair = wide[[others[0], baseline]].dropna()
        diffs[label] = (pair[others[0]] - pair[baseline])

    if len(diffs) < 2:
        return pd.DataFrame()

    shared = None
    for series in diffs.values():
        shared = series.index if shared is None else shared.intersection(series.index)
    shared = sorted(shared)
    if len(shared) < 20:
        return pd.DataFrame([{"error": f"only {len(shared)} indices shared across runs"}])

    mat = {label: series.loc[shared].to_numpy(dtype=float) for label, series in diffs.items()}
    labels = list(mat)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(n_bootstrap, len(shared)))

    boot_dz: dict[str, np.ndarray] = {}
    point_dz: dict[str, float] = {}
    for label, d in mat.items():
        sd = d.std(ddof=1)
        point_dz[label] = float(d.mean() / sd) if sd > 0 else float("nan")
        sample = d[idx]
        sd_b = sample.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            boot_dz[label] = np.where(sd_b > 0, sample.mean(axis=1) / sd_b, np.nan)

    rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            delta = boot_dz[a] - boot_dz[b]
            delta = delta[~np.isnan(delta)]
            lo, hi = np.quantile(delta, [0.025, 0.975])
            rows.append({
                "pair": f"{a} - {b}",
                f"dz_{a}": round(point_dz[a], 3),
                f"dz_{b}": round(point_dz[b], 3),
                "dz_diff": round(point_dz[a] - point_dz[b], 3),
                "diff_ci": f"[{lo:+.3f}, {hi:+.3f}]",
                "distinguishable": bool(lo > 0 or hi < 0),
            })
    out = pd.DataFrame(rows)
    return pd.DataFrame([{
        "pair": r["pair"],
        "dz_diff": r["dz_diff"],
        "diff_ci": r["diff_ci"],
        "distinguishable": r["distinguishable"],
    } for r in rows]) if not out.empty else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--metric", default=None,
                    help="force a score column (default: quality_score if present, else hit)")
    ap.add_argument("--dz-test", action="store_true",
                    help="paired bootstrap on the DIFFERENCE between rubrics' effect sizes")
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("runs", nargs="+", metavar="LABEL=PATH")
    args = ap.parse_args()

    rows = []
    runs: dict[str, pd.DataFrame] = {}
    for spec in args.runs:
        if "=" not in spec:
            raise SystemExit(f"expected LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        df = pd.read_excel(Path(path))
        if "index" not in df.columns or "model" not in df.columns:
            raise SystemExit(f"{path}: needs 'index' and 'model' columns")
        runs[label] = df
        metric = pick_metric(df, args.metric)
        row = {"rubric": label, "metric": metric}
        row.update(paired_stats(df, args.baseline, metric,
                                n_bootstrap=args.n_bootstrap))
        row.update(human_agreement(df, metric))
        rows.append(row)

    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(out.to_string(index=False))

    if args.dz_test:
        test = dz_pairwise_test(runs, args.baseline, args.metric,
                               n_bootstrap=args.n_bootstrap)
        if not test.empty:
            print("\npairwise effect-size test (resamples the shared index set):")
            print(test.to_string(index=False))
            undecided = test[~test["distinguishable"]]["pair"].tolist()
            if undecided:
                print(f"\nindistinguishable on this sample: {', '.join(undecided)}")
                print("-> pick between these on agreement with human labels, not on d_z")

    if "effect_size_dz" in out.columns and out["effect_size_dz"].notna().any():
        best = out.loc[out["effect_size_dz"].idxmax()]
        print(f"\nlargest paired effect size: {best['rubric']} "
              f"(d_z = {best['effect_size_dz']}, {best['metric']})")
        if "spearman_vs_human" in out.columns and out["spearman_vs_human"].notna().any():
            best_h = out.loc[out["spearman_vs_human"].idxmax()]
            print(f"best human agreement:       {best_h['rubric']} "
                  f"(spearman = {best_h['spearman_vs_human']})")
            if best["rubric"] != best_h["rubric"]:
                print("\nThese disagree. Prefer human agreement -- a rubric that separates the\n"
                      "models while tracking humans poorly is separating them on the wrong axis.")


if __name__ == "__main__":
    main()
