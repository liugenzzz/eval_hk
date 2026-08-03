from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .config import (
    ConfigError,
    EvalConfig,
    ModelConfig,
    is_pipeline_config,
    load_config,
    load_infer_config,
    load_pipeline_config,
)
from .pipeline import (
    PipelineError,
    run_all,
    run_conversion,
    run_evaluation,
    run_inference,
    run_sweep,
)
from .rubrics import RubricError, apply_rubric
from .run_eval import run as run_eval_stage
from .run_infer import run as run_infer_stage


SUBCOMMANDS = {"convert", "infer", "eval", "sweep", "all"}


def _model_names(value: str) -> list[str]:
    names = [item.strip() for item in str(value).split(",")]
    if not names or any(not name for name in names):
        raise argparse.ArgumentTypeError("--models requires comma-separated names")
    return names


def _rubric_names(value: str) -> list[str]:
    versions = [item.strip() for item in str(value).split(",")]
    if not versions or any(not version for version in versions):
        raise argparse.ArgumentTypeError(
            "--rubrics requires comma-separated versions"
        )
    return versions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m eval_tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert ShareGPT JSON to TSV")
    convert.add_argument("input_json")
    convert.add_argument("--config", default="pipeline.json")

    infer = subparsers.add_parser("infer", help="Run model inference")
    infer.add_argument("--config", required=True)
    infer.add_argument("--models", type=_model_names)
    infer.add_argument("--overwrite", action="store_true")
    infer.add_argument("--clean-partial", action="store_true")

    evaluate = subparsers.add_parser("eval", help="Evaluate prediction files")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--models", type=_model_names)
    evaluate.add_argument("--rubric")
    evaluate.add_argument("--out-root")
    evaluate.add_argument("--no-pairwise", action="store_true")
    evaluate.add_argument("--dry-run", action="store_true")

    sweep = subparsers.add_parser("sweep", help="Evaluate several rubrics")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--models", type=_model_names)
    sweep.add_argument("--rubrics", required=True, type=_rubric_names)
    sweep.add_argument("--out-root")
    sweep.add_argument("--no-pairwise", action="store_true")
    sweep.add_argument("--metric")
    sweep.add_argument("--n-bootstrap", type=int, default=5000)

    all_command = subparsers.add_parser("all", help="Run convert, infer, and eval")
    all_command.add_argument("--config", required=True)
    all_command.add_argument("--models", type=_model_names)
    all_command.add_argument("--overwrite", action="store_true")
    all_command.add_argument("--clean-partial", action="store_true")
    all_command.add_argument("--rubric")
    return parser


def legacy_eval_main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description="Offline multi-model Aero TSV evaluator")
    parser.add_argument("--config", required=True, help="Path to config.json or config.py")
    args = parser.parse_args(argv)
    return run_eval_stage(load_config(args.config))


def _require_pipeline(path: str) -> Any:
    if not is_pipeline_config(path):
        raise ConfigError(f"unified pipeline config required: {path}")
    return load_pipeline_config(path)


def _handle_convert(args: argparse.Namespace) -> Path:
    return run_conversion(_require_pipeline(args.config), args.input_json)


def _handle_infer(args: argparse.Namespace) -> Any:
    if is_pipeline_config(args.config):
        return run_inference(
            load_pipeline_config(args.config),
            args.models,
            overwrite=args.overwrite,
            clean_partial=args.clean_partial,
        )
    config = load_infer_config(args.config)
    if args.models is not None:
        if args.models != [config.model_name]:
            raise ConfigError(
                f"legacy infer config contains only model: {config.model_name}"
            )
    if args.overwrite or args.clean_partial:
        config = replace(
            config,
            overwrite=True if args.overwrite else config.overwrite,
            clean_partial=args.clean_partial or config.clean_partial,
        )
    return run_infer_stage(config)


def _filter_legacy_eval(config: EvalConfig, names: list[str] | None) -> EvalConfig:
    if names is None:
        return config
    by_name: dict[str, ModelConfig] = {model.name: model for model in config.models}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ConfigError(f"unknown models: {','.join(unknown)}")
    if len(set(names)) != len(names):
        raise ConfigError("duplicate names in --models")
    return replace(
        config,
        models=[by_name[name] for name in names],
        do_pairwise=config.do_pairwise and config.baseline_model in names,
    )


def _handle_eval(args: argparse.Namespace) -> Any:
    if is_pipeline_config(args.config):
        pipeline_config = load_pipeline_config(args.config)
        if not args.rubric:
            if args.out_root or args.no_pairwise or args.dry_run:
                raise ConfigError(
                    "--out-root, --no-pairwise, and --dry-run require --rubric"
                )
            return run_evaluation(pipeline_config, args.models)
        config = pipeline_config.to_eval_config(args.models)
    else:
        config = _filter_legacy_eval(load_config(args.config), args.models)
        if not args.rubric:
            if args.out_root or args.no_pairwise or args.dry_run:
                raise ConfigError(
                    "--out-root, --no-pairwise, and --dry-run require --rubric"
                )
            return run_eval_stage(config)
    derived = apply_rubric(
        config,
        args.rubric,
        out_root=args.out_root,
        no_pairwise=args.no_pairwise,
    )
    print(f"rubric      {args.rubric}")
    print(f"out_dir     {derived.out_dir}")
    print(f"do_pairwise {derived.do_pairwise}")
    print(f"fingerprint {derived.judge.fingerprint}")
    if args.dry_run:
        return derived
    return run_eval_stage(derived)


def _handle_sweep(args: argparse.Namespace) -> Any:
    return run_sweep(
        _require_pipeline(args.config),
        args.rubrics,
        model_names=args.models,
        out_root=args.out_root,
        no_pairwise=args.no_pairwise,
        metric=args.metric,
        n_bootstrap=args.n_bootstrap,
    )


def _handle_all(args: argparse.Namespace) -> Any:
    if args.rubric:
        raise PipelineError("--rubric support is added by the rubric stage")
    return run_all(
        _require_pipeline(args.config),
        args.models,
        overwrite=args.overwrite,
        clean_partial=args.clean_partial,
    )


HANDLERS: dict[str, Callable[[argparse.Namespace], Any]] = {
    "convert": _handle_convert,
    "infer": _handle_infer,
    "eval": _handle_eval,
    "sweep": _handle_sweep,
    "all": _handle_all,
}


def main(argv: list[str] | None = None) -> Any:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["-h"], ["--help"]):
        return build_parser().parse_args(arguments)
    if not arguments or arguments[0] not in SUBCOMMANDS:
        return legacy_eval_main(arguments)
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        return HANDLERS[args.command](args)
    except (ConfigError, PipelineError, RubricError) as exc:
        parser.error(str(exc))
