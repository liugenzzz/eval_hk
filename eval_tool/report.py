from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from .aggregate import (
    make_cross_tables,
    make_metric_summary,
    make_wide_summary,
    make_weighted_score_summary,
    summarize_pairwise_vs_baseline,
)
from .length_control import pairwise_length_control, pointwise_length_control


def write_reports(
    out_dir: str | Path,
    details: Iterable[pd.DataFrame],
    pairwise: pd.DataFrame | None,
    baseline_model: str,
    bootstrap_n: int,
    seed: int,
    do_length_control: bool = True,
    category_weights: dict[str, float] | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail_frames = [df.copy() for df in details if df is not None and not df.empty]
    all_details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    written: dict[str, Path] = {}

    if not all_details.empty:
        score_summary = make_weighted_score_summary(all_details, category_weights or {})
        if not score_summary.empty:
            score_path = out / "score_summary.csv"
            score_summary.to_csv(score_path, index=False, encoding="utf-8-sig")
            written["score_summary.csv"] = score_path

        long_summary = make_metric_summary(all_details, bootstrap_n=bootstrap_n, seed=seed)
        wide_summary = make_wide_summary(long_summary)
        if do_length_control:
            vqa = all_details[all_details["dataset"] == "vqa"].copy()
            if not vqa.empty:
                lc = pointwise_length_control(vqa, baseline_model)
                if not lc.empty and not wide_summary.empty:
                    wide_summary = wide_summary.merge(lc, on="model", how="left")
        summary_csv = out / "report_summary.csv"
        summary_json = out / "report_summary.json"
        wide_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
        summary_json.write_text(wide_summary.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")
        written["report_summary.csv"] = summary_csv
        written["report_summary.json"] = summary_json

        long_path = out / "report_summary_long.csv"
        long_summary.to_csv(long_path, index=False, encoding="utf-8-sig")
        written["report_summary_long.csv"] = long_path

        for model, table in make_cross_tables(all_details).items():
            path = out / f"cross_{_safe_name(model)}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
            written[f"cross_{model}.csv"] = path

        for (model, dataset), df in all_details.groupby(["model", "dataset"]):
            path = out / f"detail_{_safe_name(model)}_{dataset}.xlsx"
            _detail_without_image(df).to_excel(path, index=False)
            written[f"detail_{model}_{dataset}.xlsx"] = path

        # All models' judge results side by side in one file, for cross-model inspection.
        combined_path = out / "judge_detail_all.xlsx"
        combined_cols = [
            col
            for col in (
                "model",
                "dataset",
                "index",
                "category",
                "question",
                "answer",
                "prediction",
                "hit",
                "quality_score",
                "param_score",
                "fact_score",
                "visual_score",
                "fabrication_score",
                "style_score",
                "judge_reason",
            )
            if col in all_details.columns
        ]
        _detail_without_image(all_details[combined_cols]).to_excel(combined_path, index=False)
        written["judge_detail_all.xlsx"] = combined_path

    if pairwise is not None and not pairwise.empty:
        pairwise_summary = summarize_pairwise_vs_baseline(pairwise, bootstrap_n=bootstrap_n, seed=seed)
        if do_length_control:
            lc_pairwise = pairwise_length_control(pairwise)
            if not lc_pairwise.empty and not pairwise_summary.empty:
                pairwise_summary = pairwise_summary.merge(lc_pairwise, on="model", how="left")
        pairwise_path = out / "pairwise_vs_baseline.csv"
        pairwise_summary.to_csv(pairwise_path, index=False, encoding="utf-8-sig")
        written["pairwise_vs_baseline.csv"] = pairwise_path
        detail_path = out / "pairwise_vs_baseline_detail.xlsx"
        _detail_without_image(pairwise).to_excel(detail_path, index=False)
        written["pairwise_vs_baseline_detail.xlsx"] = detail_path

    manifest = out / "manifest.json"
    manifest.write_text(
        json.dumps({k: str(v) for k, v in written.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    written["manifest.json"] = manifest
    return written


def _safe_name(name: object) -> str:
    return str(name).replace("/", "_").replace("\\", "_").replace(":", "_")


def _detail_without_image(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(columns=[col for col in data.columns if col == "image"], errors="ignore")
