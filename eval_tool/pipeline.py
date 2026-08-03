from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from compare_rubrics import compare_runs
from .config import PipelineConfig
from .convert_vqa_json import convert as convert_vqa
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


def _require_existing_tsvs(config: PipelineConfig) -> None:
    for dataset_key in config.enabled_datasets:
        dataset_name = config.datasets[dataset_key]
        path = config.tsv_dir / f"{dataset_name}.tsv"
        if not path.is_file():
            raise PipelineError(
                f"missing TSV and convert.input_json is unset: {dataset_name}"
            )


def run_all(
    config: PipelineConfig,
    model_names: list[str] | tuple[str, ...] | None = None,
    generator_factory: Callable[[str], Any] | None = None,
    overwrite: bool = False,
    clean_partial: bool = False,
) -> dict[str, Any]:
    converted: Path | None = None
    if config.convert_input is not None:
        converted = run_conversion(config)
    else:
        _require_existing_tsvs(config)
    inferred = run_inference(
        config,
        model_names,
        generator_factory,
        overwrite,
        clean_partial,
    )
    evaluated = run_evaluation(config, model_names)
    return {"convert": converted, "infer": inferred, "eval": evaluated}
