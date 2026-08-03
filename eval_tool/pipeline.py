from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig
from .convert_vqa_json import convert as convert_vqa
from .run_eval import run as run_eval
from .run_infer import run as run_infer


class PipelineError(RuntimeError):
    pass


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
