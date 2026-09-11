from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from compare_rubrics import compare_runs
from .artifacts import prediction_stems
from .config import DerivePlan, PipelineConfig
from .convert_vqa_json import convert as convert_vqa
from .derive import (
    derive_model_history,
    derive_question_perturbation,
    derive_reverse_consistency,
    load_predictions,
    write_jsonl,
)
from .eval_set import load_records as load_eval_records, sample_records
from .io import truth_path
from .phrase_pool import PhrasePool
from .run_eval import run as run_eval
from .run_infer import run as run_infer
from .rubrics import apply_rubric, validate_rubric_list


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class SweepResult:
    reports: dict[str, dict[str, Path]]
    comparison_path: Path


def run_conversion(
    config: PipelineConfig,
    input_json: str | Path | None = None,
) -> Path:
    source = Path(input_json) if input_json is not None else config.convert_input
    if source is None:
        raise PipelineError("convert.input_json is unset and no input JSON was provided")
    return convert_vqa(source, config.tsv_dir)


def run_inference(
    config: PipelineConfig,
    model_names: list[str] | tuple[str, ...] | None = None,
    generator_factory: Callable[[str], Any] | None = None,
    overwrite: bool = False,
    clean_partial: bool = False,
) -> dict[str, dict[str, Path]]:
    written: dict[str, dict[str, Path]] = {}
    infer_configs = config.to_infer_configs(
        model_names,
        overwrite=overwrite,
        clean_partial=clean_partial,
    )
    for infer_config in infer_configs:
        generator = (
            None
            if generator_factory is None
            else generator_factory(infer_config.model_name)
        )
        written[infer_config.model_name] = run_infer(
            infer_config, generator=generator
        )
    return written


def run_evaluation(
    config: PipelineConfig,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    return run_eval(config.to_eval_config(model_names))


def run_rubric_evaluation(
    config: PipelineConfig,
    version: str,
    model_names: list[str] | tuple[str, ...] | None = None,
    out_root: str | Path | None = None,
    no_pairwise: bool = False,
) -> dict[str, Path]:
    eval_config = apply_rubric(
        config.to_eval_config(model_names),
        version,
        out_root=out_root,
        no_pairwise=no_pairwise,
    )
    return run_eval(eval_config)


def run_sweep(
    config: PipelineConfig,
    rubrics: list[str] | tuple[str, ...],
    model_names: list[str] | tuple[str, ...] | None = None,
    out_root: str | Path | None = None,
    no_pairwise: bool = False,
    metric: str | None = None,
    n_bootstrap: int = 5000,
) -> SweepResult:
    versions = validate_rubric_list(rubrics)
    selected = config.select_models(model_names)
    if any(model.scored_paths for model in selected):
        raise PipelineError(
            "sweep cannot use models[].scored because its rubric is unverifiable"
        )
    reports: dict[str, dict[str, Path]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for version in versions:
        eval_config = apply_rubric(
            config.to_eval_config(model_names),
            version,
            out_root=out_root,
            no_pairwise=no_pairwise,
        )
        reports[version] = run_eval(eval_config)
        detail = reports[version].get("judge_detail_all.xlsx")
        if detail is None or not Path(detail).is_file():
            raise PipelineError(
                f"rubric {version} did not produce judge_detail_all.xlsx"
            )
        frames[version] = pd.read_excel(detail, dtype={"index": str})
    comparison = compare_runs(
        frames,
        baseline=config.baseline_model,
        metric=metric,
        n_bootstrap=n_bootstrap,
        seed=config.seed,
    )
    comparison_root = (
        config.out_dir.parent if out_root is None else Path(out_root)
    )
    comparison_root.mkdir(parents=True, exist_ok=True)
    comparison_path = comparison_root / "rubric_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    return SweepResult(reports=reports, comparison_path=comparison_path)


def _require_existing_truth(config: PipelineConfig, dataset_keys: list[str]) -> None:
    """真值文件在不在。**jsonl 也算** —— 目标检测那条路读的就是 test.jsonl，
    以前这里只找 .tsv，导致 ``eval_tool all`` 在第一步就报 missing TSV。

    派生集不在检查范围里：它们要等主线推理跑完才存在，调用方把它们排除掉再传进来。
    """
    for dataset_key in dataset_keys:
        dataset_name = config.datasets[dataset_key]
        # truth_path 优先给 jsonl，两个都没有时回落到 .tsv 那个（不存在的）路径。
        if not truth_path(config.tsv_dir, dataset_name).is_file():
            raise PipelineError(
                f"missing TSV/JSONL and convert.input_json is unset: {dataset_name}"
            )


def run_derive(
    config: PipelineConfig,
    model_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """按配置里的 ``derive`` 块造派生评估集。

    纯函数，不用 GPU。造出来的就是普通 jsonl，下一趟 infer 当普通数据集跑，
    断点续传和缓存全部免费继承 —— 推理引擎一个字都不用改。
    """
    if not config.derive_plans:
        return {}
    stems = prediction_stems(config.datasets)
    selected = {model.name for model in config.select_models(model_names)}
    written: dict[str, Path] = {}
    for plan in config.derive_plans:
        if plan.needs_predictions and plan.from_model not in selected:
            print(
                f"[derive] 跳过 {plan.dataset}：这一趟没跑 {plan.from_model}",
                flush=True,
            )
            continue
        # 主线抽了样，派生集要抽同一批 —— 否则问法扰动会拿全量样本去问一遍，
        # 而模型只答过抽中的那 N 条，剩下的全是缺预测。
        source_params = config.dataset_params.get(plan.source) or {}
        sample_n = source_params.get("sample_n")
        records = sample_records(
            load_eval_records(truth_path(config.tsv_dir, config.datasets[plan.source])),
            int(sample_n) if sample_n else None,
            int(source_params.get("sample_seed", 42)),
        )
        out_path = config.tsv_dir / f"{config.datasets[plan.dataset]}.jsonl"
        derived = _derive_one(config, plan, records, stems)
        if not derived:
            sampled = bool(sample_n) or plan.sample_ratio < 1.0
            if sampled:
                # 抽了样还一条都派生不出来是正常的：派生要的那几个 task_type 可能
                # 一条都没抽中。跳过这一个数据集，别把整趟跑挂掉。
                print(
                    f"[derive] {plan.dataset}: 抽样之后没有可派生的样本，跳过。"
                    "要评它就调大 sample.n / sample_ratio。",
                    flush=True,
                )
                continue
            # 没抽样却派生出空的，那是 from/tasks/pools 配错了。写出一份空评估集
            # 会让下一趟推理正常跑完，然后报表上多一格「样本不足」—— 而那一格
            # 实际上是这里配错了。
            raise PipelineError(
                f"derive {plan.dataset} 产出为空。检查 from / tasks / pools 配得对不对。"
            )
        written[plan.dataset] = write_jsonl(derived, out_path)
        print(f"[derive] {plan.dataset}: {len(derived)} 条 -> {out_path}", flush=True)
    return written


def _derive_one(
    config: PipelineConfig,
    plan: DerivePlan,
    records: list[dict[str, Any]],
    stems: dict[str, str],
) -> list[dict[str, Any]]:
    tasks = plan.tasks or None
    if plan.mode == "question-perturbation":
        return derive_question_perturbation(
            records,
            pools={key: PhrasePool.load(path) for key, path in plan.pools.items()},
            variants=plan.variants,
            sample_ratio=plan.sample_ratio,
            tasks=tasks,
        )
    # 主线那几个数据集共用一份评估集，预测却分散在各自的文件里（一条 inventory_locate
    # 的三轮就在三个数据集里）。把这个模型所有非派生数据集的预测都喂进去，让 derive
    # 自己按 index 取它要的那几轮。
    derived_keys = {item.dataset for item in config.derive_plans}
    preds = [
        config.artifacts.prediction(plan.from_model, stems[key])
        for key in config.enabled_datasets
        if key not in derived_keys
    ]
    existing = [path for path in preds if path.is_file()]
    if not existing:
        raise PipelineError(
            f"derive {plan.dataset} 要 {plan.from_model} 的预测，但一个都没找到。"
            f"先跑 infer，或者在 models[].pred 里指到已有的预测文件。"
        )
    predictions = load_predictions(existing)
    if plan.mode == "model-history":
        return derive_model_history(
            records,
            predictions,
            target_turns=plan.target_turns,
            sample_ratio=plan.sample_ratio,
        )
    pool_path = plan.pools.get("region_identify") or plan.pools.get("default")
    return derive_reverse_consistency(
        records,
        predictions,
        question_pool=PhrasePool.load(pool_path),
        scale=plan.scale,
        tasks=tasks,
    )


def run_all(
    config: PipelineConfig,
    model_names: list[str] | tuple[str, ...] | None = None,
    generator_factory: Callable[[str], Any] | None = None,
    overwrite: bool = False,
    clean_partial: bool = False,
    rubric: str | None = None,
) -> dict[str, Any]:
    derived_keys = {plan.dataset for plan in config.derive_plans}
    mainline = [key for key in config.enabled_datasets if key not in derived_keys]
    derived = [key for key in config.enabled_datasets if key in derived_keys]

    converted: Path | None = None
    if config.convert_input is not None:
        converted = run_conversion(config)
    else:
        # 派生集这会儿还不存在，别拿它们卡前置检查。
        _require_existing_truth(config, mainline)

    inferred = run_inference(
        replace(config, enabled_datasets=mainline),
        model_names,
        generator_factory,
        overwrite,
        clean_partial,
    )
    # 派生集必须造在两趟推理中间：它们的内容就是模型自己在第一趟里的输出。
    derived_files = run_derive(config, model_names) if derived else {}
    produced = [key for key in derived if key in derived_files]
    if produced:
        second = run_inference(
            replace(config, enabled_datasets=produced),
            model_names,
            generator_factory,
            overwrite,
            clean_partial,
        )
        for model_name, paths in second.items():
            inferred.setdefault(model_name, {}).update(paths)

    # 没造出来的派生集要从评估里摘掉 —— 它的 jsonl 根本不存在，留着会让评估直接
    # 报文件找不到，而它没造出来的原因（抽样抽空了）本身是正常的。
    skipped = [key for key in derived if key not in derived_files]
    for_eval = (
        replace(config, enabled_datasets=[
            key for key in config.enabled_datasets if key not in skipped
        ])
        if skipped
        else config
    )
    evaluated = (
        run_evaluation(for_eval, model_names)
        if rubric is None
        else run_rubric_evaluation(for_eval, rubric, model_names)
    )
    return {
        "convert": converted,
        "infer": inferred,
        "derive": derived_files,
        "eval": evaluated,
    }
