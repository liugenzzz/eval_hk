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
    summarize_paired_diff,
    summarize_pairwise_vs_baseline,
)
from .breakdown import (
    EmptyCells,
    DimSpec,
    default_dims,
    make_breakdown,
    make_failure_buckets,
    make_weighted_total,
    parse_dims,
)
from .length_control import pairwise_length_control, pointwise_length_control

# 达标率之外还值得按维度拆的连续量。四点偏差和 IoU 是连续值，只报一个达标率会丢掉
# 「差多少」的信息 —— 达标率一样的两个模型，偏差均值可以差一倍。
BREAKDOWN_METRICS = ("hit", "localized", "dev_mean4_pct", "iou", "format_ok", "task_bleed")


def write_reports(
    out_dir: str | Path,
    details: Iterable[pd.DataFrame],
    pairwise: pd.DataFrame | None,
    baseline_model: str,
    bootstrap_n: int,
    seed: int,
    do_length_control: bool = True,
    category_weights: dict[str, float] | None = None,
    report_dims: list[DimSpec] | None = None,
    empty_cells: EmptyCells | None = None,
    dataset_kinds: dict[str, str] | None = None,
    dataset_engines: dict[str, str] | None = None,
    dataset_weights: dict[str, float] | None = None,
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

        written.update(
            _write_breakdowns(
                out,
                all_details,
                baseline_model=baseline_model,
                bootstrap_n=bootstrap_n,
                seed=seed,
                report_dims=report_dims,
                empty_cells=empty_cells,
                dataset_kinds=dataset_kinds,
                dataset_engines=dataset_engines,
                dataset_weights=dataset_weights,
            )
        )

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
                "accuracy_score",
                "equipment_correct",
                "applicable_dims",
                "failure_type",
                "param_score",
                "fact_score",
                "visual_score",
                "fabrication_score",
                "style_score",
                "relevance_score",
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


def _write_breakdowns(
    out: Path,
    all_details: pd.DataFrame,
    *,
    baseline_model: str,
    bootstrap_n: int,
    seed: int,
    report_dims: list[DimSpec] | None,
    empty_cells: EmptyCells | None,
    dataset_kinds: dict[str, str] | None,
    dataset_engines: dict[str, str] | None,
    dataset_weights: dict[str, float] | None,
) -> dict[str, Path]:
    """§17 的拆分表、§16.2 的配对区间、验收总分。

    这几张表才是这次评估的产出。report_summary 那几张是旧通路留下的总览，一个总分
    回答不了「哪一档、哪一类、哪种尺寸还不行」。
    """
    written: dict[str, Path] = {}
    dims = report_dims if report_dims is not None else default_dims()
    breakdown = make_breakdown(
        all_details,
        dims,
        metrics=[m for m in BREAKDOWN_METRICS if m in all_details.columns],
        kinds=dataset_kinds or {},
        empty_cells=empty_cells,
        bootstrap_n=bootstrap_n,
        seed=seed,
    )
    if not breakdown.empty:
        path = out / "breakdown.csv"
        breakdown.to_csv(path, index=False, encoding="utf-8-sig")
        written["breakdown.csv"] = path

    buckets = make_failure_buckets(all_details)
    if not buckets.empty:
        path = out / "failure_buckets.csv"
        buckets.to_csv(path, index=False, encoding="utf-8-sig")
        written["failure_buckets.csv"] = path

    if dataset_engines:
        total = make_weighted_total(all_details, dataset_weights or {}, dataset_engines)
        if not total.empty:
            path = out / "acceptance_score.csv"
            total.to_csv(path, index=False, encoding="utf-8-sig")
            written["acceptance_score.csv"] = path

    paired_frames = [
        summarize_paired_diff(all_details, baseline_model, metric_col=metric,
                              group_cols=["dataset"], bootstrap_n=bootstrap_n, seed=seed)
        for metric in BREAKDOWN_METRICS
        if metric in all_details.columns
    ]
    paired_frames = [frame for frame in paired_frames if not frame.empty]
    if paired_frames:
        path = out / "paired_diff_vs_baseline.csv"
        pd.concat(paired_frames, ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
        written["paired_diff_vs_baseline.csv"] = path
    return written
