from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cache import JsonlCache
from .config import EvalConfig, load_config
from .io import align_truth_and_prediction, image_map_from_truth, load_prediction_file, load_truth_dataset, normalize_index, read_table
from .judge import JudgeClient
from .metrics_text import aux_metrics
from .report import write_reports
from .score_mcq import score_choice_dataframe
from .score_vqa import score_pairwise_vs_baseline, score_pointwise_vqa


def run(config: EvalConfig) -> dict[str, Path]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    truth = {
        key: load_truth_dataset(config.tsv_dir, name)
        for key, name in config.datasets.items()
        if key in config.enabled_datasets
    }
    image_map = image_map_from_truth(truth.get("vqa", pd.DataFrame()))
    judge_client = JudgeClient(config.judge)
    pointwise_cache = JsonlCache(config.cache_dir / "judge_cache_pointwise.jsonl", ("model", "dataset", "index"))
    pairwise_cache = JsonlCache(config.cache_dir / "judge_cache_pairwise.jsonl", ("model_A", "model_B", "index", "direction"))
    details: list[pd.DataFrame] = []
    vqa_by_model: dict[str, pd.DataFrame] = {}
    warnings: list[str] = []

    vqa_truth = truth.get("vqa")
    if vqa_truth is not None and not vqa_truth.empty:
        missing_idx = [
            str(row["index"])
            for _, row in vqa_truth.iterrows()
            if str(row.get("index", "")) not in image_map
        ]
        if missing_idx:
            msg = (
                f"[warn] vqa 真值中有 {len(missing_idx)}/{len(vqa_truth)} 行没有可用图片（image 列为空/NaN），"
                f"这些行判分时将不带图。示例 index: {missing_idx[:10]}"
            )
            print(msg, flush=True)
            warnings.append(msg)

    for model in config.models:
        for dataset_key in ("mcq", "judge", "vqa"):
            if dataset_key not in config.enabled_datasets:
                continue
            scored_path = model.scored_path_for(dataset_key)
            if scored_path:
                if not Path(scored_path).exists():
                    warnings.append(f"[skip] {model.name} {dataset_key} scored file not found: {scored_path}")
                    continue
                # normalize_index keeps "index" a str, matching every other dataframe in this
                # pipeline -- plain pd.read_excel would infer numeric-looking indices as int64
                # and blow up the pairwise merge ("merge on object and int64 columns").
                scored = normalize_index(read_table(scored_path))
                scored["model"] = model.name
                scored["dataset"] = dataset_key
                if dataset_key == "vqa":
                    vqa_by_model[model.name] = scored
                details.append(scored)
                continue
            pred_path = model.path_for(dataset_key)
            if not pred_path:
                warnings.append(f"[skip] {model.name} has no {dataset_key} prediction path")
                continue
            if not Path(pred_path).exists():
                warnings.append(f"[skip] {model.name} {dataset_key} prediction file not found: {pred_path}")
                continue
            aligned = align_truth_and_prediction(truth[dataset_key], load_prediction_file(pred_path))
            data = aligned.data
            data.insert(0, "model", model.name)
            data.insert(1, "dataset", dataset_key)
            if aligned.missing_count:
                warnings.append(f"[warn] {model.name} {dataset_key}: {aligned.missing_count} missing predictions")
            if not aligned.extra_predictions.empty:
                extra_path = config.out_dir / f"extra_predictions_{model.name}_{dataset_key}.csv"
                aligned.extra_predictions.to_csv(extra_path, index=False, encoding="utf-8-sig")
                warnings.append(f"[warn] {model.name} {dataset_key}: extra predictions written to {extra_path}")

            if dataset_key in {"mcq", "judge"}:
                scored = score_choice_dataframe(data, dataset=dataset_key)
            else:
                if config.do_pointwise:
                    scored = score_pointwise_vqa(
                        data,
                        model.name,
                        judge_client,
                        cache=pointwise_cache,
                        workers=config.max_workers,
                        image_map=image_map,
                    )
                else:
                    scored = data.copy()
                    scored["hit"] = pd.NA
                    aux = [aux_metrics(row.get("answer", ""), row.get("prediction", "")) for _, row in scored.iterrows()]
                    for col in ("bleu1", "bleu2", "rouge_l", "pred_len"):
                        scored[col] = [m[col] for m in aux]
                vqa_by_model[model.name] = scored
            details.append(scored)

    pairwise = pd.DataFrame()
    if config.do_pairwise and config.baseline_model in vqa_by_model and len(vqa_by_model) > 1:
        baseline_df = vqa_by_model[config.baseline_model]
        pairwise_frames = []
        for model_name, model_df in vqa_by_model.items():
            if model_name == config.baseline_model:
                continue
            pairwise_frames.append(
                score_pairwise_vs_baseline(
                    model_df,
                    baseline_df,
                    model_name,
                    config.baseline_model,
                    judge_client,
                    cache=pairwise_cache,
                    workers=config.max_workers,
                    image_map=image_map,
                )
            )
        if pairwise_frames:
            pairwise = pd.concat(pairwise_frames, ignore_index=True)
    elif config.do_pairwise:
        warnings.append("[skip] pairwise requires baseline model and at least one non-baseline VQA result")

    written = write_reports(
        config.out_dir,
        details,
        pairwise,
        baseline_model=config.baseline_model,
        bootstrap_n=config.bootstrap_n,
        seed=config.seed,
        do_length_control=config.do_length_control,
        category_weights=config.category_weights,
    )
    if warnings:
        warn_path = config.out_dir / "warnings.log"
        warn_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        written["warnings.log"] = warn_path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline multi-model Aero TSV evaluator")
    parser.add_argument("--config", required=True, help="Path to config.json or config.py")
    args = parser.parse_args()
    written = run(load_config(args.config))
    print("Written files:")
    for name, path in written.items():
        print(f"- {name}: {path}")
    if "score_summary.csv" in written:
        print("\n=== Score Summary (weighted by category_weights) ===")
        print(pd.read_csv(written["score_summary.csv"]).to_string(index=False))


if __name__ == "__main__":
    main()
