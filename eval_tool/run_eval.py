from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import scorers
from .cache import JsonlCache
from .breakdown import EmptyCells, parse_dims
from .compliance import attach_compliance
from .eval_set import sha256_of
from .fingerprint import RunFingerprint, check_comparable, dataset_shas, stamp
from .scale import scale_of
from .config import EvalConfig, load_config
from .io import align_truth_and_prediction, image_map_from_truth, load_prediction_file, load_truth_dataset, normalize_index, read_table, truth_path
from .judge import JudgeClient
from .report import write_reports
from .score_vqa import score_pairwise_vs_baseline


def _scoring_plan(config: EvalConfig) -> list[tuple[str, scorers.ScorerSpec]]:
    """按配置顺序解析出 (数据集键, 打分器)，起跑前就把未知 kind 报出来。

    以前这里是写死的 ("mcq", "judge", "vqa") 三元组循环。现在数据集想叫什么名字
    都行，打分方式由它自己声明的 kind 决定；enabled_datasets 只做过滤。
    kind 查不到就直接抛 —— 跑完两小时推理再发现打分器不存在，代价太大。
    """
    plan: list[tuple[str, scorers.ScorerSpec]] = []
    for dataset_key in config.enabled_datasets:
        if dataset_key not in config.datasets:
            continue
        plan.append((dataset_key, scorers.get(config.kind_for(dataset_key))))
    return plan


def run(config: EvalConfig) -> dict[str, Path]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    plan = _scoring_plan(config)
    truth = {
        dataset_key: load_truth_dataset(
            config.tsv_dir,
            config.datasets[dataset_key],
            config.params_for(dataset_key),
            # 代码打分器不看图。判分时为了几个纯代码指标把整批图读进内存没有道理，
            # 而推理端读同一份配置时一律要图（见 load_truth_dataset）。
            need_images=spec.needs_judge,
        )
        for dataset_key, spec in plan
    }
    # 图片按数据集各建一张表：不同数据集的 index 是各自编号的，合成一张会串图。
    image_maps = {key: image_map_from_truth(frame) for key, frame in truth.items()}
    # §13 评估集冻结：抽样一次后固化，每份结果记下它评的是哪一批样本。
    fingerprint = RunFingerprint(
        eval_set_sha=dataset_shas(
            {key: sha256_of(truth_path(config.tsv_dir, config.datasets[key])) for key, _ in plan}
        ),
        profile_version=config.profile_version,
        rubric_version=config.judge.fingerprint,
        judge_model=config.judge.model,
    )
    judge_client = JudgeClient(config.judge)
    # judge_fp first: it is a hash of the judge model, temperature and both prompt texts,
    # so switching rubric versions misses the cache and re-scores instead of silently
    # replaying verdicts from the previous prompt. Rows written before this existed have
    # no judge_fp and fail to match, which is the intended behaviour -- they were scored
    # under a different rubric and are not comparable.
    pointwise_cache = JsonlCache(
        config.cache_dir / "judge_cache_pointwise.jsonl",
        ("judge_fp", "model", "dataset", "index"),
    )
    pairwise_cache = JsonlCache(
        config.cache_dir / "judge_cache_pairwise.jsonl",
        ("judge_fp", "model_A", "model_B", "index", "direction"),
    )
    print(f"[judge] model={config.judge.model} fingerprint={config.judge.fingerprint}", flush=True)
    details: list[pd.DataFrame] = []
    # dataset_key -> model_name -> 打过分的表，供 pairwise 取用
    scored_by_dataset: dict[str, dict[str, pd.DataFrame]] = {}
    warnings: list[str] = []

    for dataset_key, spec in plan:
        if not spec.needs_judge:
            continue
        dataset_truth = truth[dataset_key]
        image_map = image_maps[dataset_key]
        missing_idx = [
            str(row["index"])
            for _, row in dataset_truth.iterrows()
            if str(row.get("index", "")) not in image_map
        ]
        if missing_idx:
            msg = (
                f"[warn] {dataset_key} 真值中有 {len(missing_idx)}/{len(dataset_truth)} 行没有可用图片（image 列为空/NaN），"
                f"这些行判分时将不带图。示例 index: {missing_idx[:10]}"
            )
            print(msg, flush=True)
            warnings.append(msg)

    for model in config.models:
        for dataset_key, spec in plan:
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
                scored = _with_compliance(scored, spec, config, dataset_key)
                # 复用的 scored 文件如果已经带指纹就保留它 —— 覆盖掉就等于把
                # 「它是用旧口径打的」抹了，校验也就查不出来了。
                scored = stamp(scored, fingerprint)
                if spec.pairwise:
                    scored_by_dataset.setdefault(dataset_key, {})[model.name] = scored
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

            scored = spec.score(
                data,
                scorers.ScoringContext(
                    dataset_key=dataset_key,
                    kind=spec.kind,
                    model_name=model.name,
                    params=config.params_for(dataset_key),
                    base_dir=config.base_dir,
                    judge_client=judge_client,
                    judge_cache=pointwise_cache,
                    image_map=image_maps[dataset_key],
                    do_pointwise=config.do_pointwise,
                    max_workers=config.max_workers,
                ),
            )
            scored = _with_compliance(scored, spec, config, dataset_key)
            scored = stamp(scored, fingerprint)
            if spec.pairwise:
                scored_by_dataset.setdefault(dataset_key, {})[model.name] = scored
            details.append(scored)

    if details:
        check_comparable(pd.concat(details, ignore_index=True))

    pairwise = _run_pairwise(
        config,
        plan,
        scored_by_dataset,
        image_maps,
        judge_client,
        pairwise_cache,
        warnings,
    )

    written = write_reports(
        config.out_dir,
        details,
        pairwise,
        baseline_model=config.baseline_model,
        bootstrap_n=config.bootstrap_n,
        seed=config.seed,
        do_length_control=config.do_length_control,
        category_weights=config.category_weights,
        report_dims=parse_dims(config.report_dims) if config.report_dims else None,
        empty_cells=EmptyCells.parse(config.empty_cells),
        dataset_kinds={key: spec.kind for key, spec in plan},
        dataset_engines={key: spec.engine for key, spec in plan},
        dataset_weights=config.dataset_weights,
        chain_decay_pairs=config.chain_decay_pairs,
        min_category_n=config.min_category_n,
    )
    fingerprint_path = config.out_dir / "run_fingerprint.json"
    fingerprint_path.write_text(
        json.dumps(
            {**fingerprint.to_dict(), "profile_name": config.profile_name,
             "models": [model.name for model in config.models],
             "datasets": {key: config.datasets[key] for key, _ in plan}},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    written["run_fingerprint.json"] = fingerprint_path

    if warnings:
        warn_path = config.out_dir / "warnings.log"
        warn_path.write_text("\n".join(warnings) + "\n", encoding="utf-8")
        written["warnings.log"] = warn_path
    return written


def _with_compliance(
    scored: pd.DataFrame, spec: scorers.ScorerSpec, config: EvalConfig, dataset_key: str
) -> pd.DataFrame:
    """§10 的格式合规率 / 任务串味率 / 截断嫌疑对每个数据集都算。

    这三个降到接近 0 是 SFT 最先体现的效果，也最能早期发现训练配置有问题 —— 只给
    画框那几个数据集算就看不见「问描述吐坐标」这种串味。
    """
    scale, _ = scale_of(scored, config.params_for(dataset_key))
    return attach_compliance(scored, spec.answer_form, scale=scale)


def _run_pairwise(
    config: EvalConfig,
    plan: list[tuple[str, scorers.ScorerSpec]],
    scored_by_dataset: dict[str, dict[str, pd.DataFrame]],
    image_maps: dict[str, dict[str, str]],
    judge_client: JudgeClient,
    pairwise_cache: JsonlCache,
    warnings: list[str],
) -> pd.DataFrame:
    if not config.do_pairwise:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    ran_any = False
    for dataset_key, spec in plan:
        if not spec.pairwise:
            continue
        by_model = scored_by_dataset.get(dataset_key, {})
        if config.baseline_model not in by_model or len(by_model) < 2:
            continue
        ran_any = True
        baseline_df = by_model[config.baseline_model]
        for model_name, model_df in by_model.items():
            if model_name == config.baseline_model:
                continue
            frame = score_pairwise_vs_baseline(
                model_df,
                baseline_df,
                model_name,
                config.baseline_model,
                judge_client,
                cache=pairwise_cache,
                workers=config.max_workers,
                image_map=image_maps[dataset_key],
            )
            # 多个数据集都做 pairwise 时，没有这一列的行是分不开的。
            frame.insert(1, "dataset", dataset_key)
            frames.append(frame)
    if not ran_any:
        warnings.append(
            "[skip] pairwise requires a pairwise-capable dataset with the baseline model "
            "and at least one non-baseline model scored"
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
