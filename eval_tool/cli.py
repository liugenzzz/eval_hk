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
from .derive import (
    derive_model_history,
    derive_question_perturbation,
    derive_reverse_consistency,
    load_predictions,
    write_jsonl,
)
from .dpo_config import load_dpo_config
from .eval_set import load_records as load_eval_records
from .phrase_pool import PhrasePool
from .dpo_pipeline import DpoPipelineError, run_build_dpo
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


SUBCOMMANDS = {"check", "convert", "infer", "eval", "sweep", "all", "build-dpo", "derive"}


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

    check = subparsers.add_parser(
        "check", help="开跑前体检：路径、模型、评估集、裁判一次全查（不跑推理）"
    )
    check.add_argument("--config", required=True)

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

    derive = subparsers.add_parser(
        "derive",
        help="Derive a follow-up eval set from a model's own predictions "
             "(model history / reverse consistency / question perturbation)",
    )
    derive.add_argument("--eval-set", required=True, help="原始评估集 jsonl")
    derive.add_argument("--out", required=True, help="派生集写到哪里（jsonl）")
    derive.add_argument(
        "--mode", required=True,
        choices=["model-history", "reverse-consistency", "question-perturbation"],
    )
    derive.add_argument(
        "--pred", dest="preds", action="append", default=[],
        help="模型预测文件；可重复（拼模型历史要多轮的预测，分散在多个数据集里）",
    )
    derive.add_argument("--pool", dest="pools", action="append", default=[],
                        help="问法池，形如 task=path 或 default=path；可重复")
    derive.add_argument("--sample-ratio", type=float, default=1.0,
                        help="确定性抽样比例（链路衰减率是统计量，默认那一档是 0.3）")
    derive.add_argument("--variants", type=int, default=3)
    derive.add_argument("--tasks", help="逗号分隔，只对这些 task_type 派生")
    derive.add_argument("--target-turn", dest="target_turns", action="append", default=[],
                        help="形如 inventory_locate=3；不写就取每条的最后一轮")
    derive.add_argument("--scale", type=int, default=1000)

    build_dpo = subparsers.add_parser(
        "build-dpo", help="Build a DPO dataset directly from JSON/JSONL"
    )
    build_dpo.add_argument("--config", required=True)
    build_dpo.add_argument(
        "--input",
        dest="inputs",
        action="append",
        help="Replace the configured inputs; repeat for several files",
    )
    build_dpo.add_argument("--dry-run", action="store_true")
    build_dpo.add_argument("--overwrite", action="store_true")
    build_dpo.add_argument("--clean-partial", action="store_true")
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
    # The new subcommand is resume-safe even when adapting an old infer schema.
    # The standalone ``python -m eval_tool.run_infer`` entry point still preserves
    # the legacy config semantics for R9 compatibility.
    config = replace(
        config,
        resume=True,
        overwrite=args.overwrite,
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
    return run_all(
        _require_pipeline(args.config),
        args.models,
        overwrite=args.overwrite,
        clean_partial=args.clean_partial,
        rubric=args.rubric,
    )


def _handle_build_dpo(args: argparse.Namespace) -> Any:
    config = load_dpo_config(
        args.config,
        input_overrides=args.inputs,
        invocation_dir=Path.cwd(),
    )
    return run_build_dpo(
        config,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        clean_partial=args.clean_partial,
    )


def _key_value_pairs(items: list[str], what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        key, sep, value = str(item).partition("=")
        if not sep or not key.strip() or not value.strip():
            raise ConfigError(f"{what} 要写成 key=value，得到：{item}")
        out[key.strip()] = value.strip()
    return out


def _handle_derive(args: argparse.Namespace) -> Path:
    """派生集：读原始评估集 + 模型自己的预测，产出一份新的 jsonl。

    这一步是纯函数，不需要 GPU，也不碰推理引擎 —— 产出的 jsonl 当成普通数据集再跑
    一次 infer 即可，断点续传和缓存全部免费继承。
    """
    records = load_eval_records(args.eval_set)
    tasks = [t.strip() for t in str(args.tasks).split(",")] if args.tasks else None

    if args.mode == "model-history":
        if not args.preds:
            raise ConfigError("model-history 需要 --pred（前面几轮的模型预测）")
        target_turns = {k: int(v) for k, v in _key_value_pairs(args.target_turns, "--target-turn").items()}
        derived = derive_model_history(
            records,
            load_predictions(args.preds),
            target_turns=target_turns,
            sample_ratio=args.sample_ratio,
        )
    elif args.mode == "reverse-consistency":
        if not args.preds:
            raise ConfigError("reverse-consistency 需要 --pred（正向那一轮的框）")
        pools = _key_value_pairs(args.pools, "--pool")
        pool_path = pools.get("region_identify") or pools.get("default")
        if not pool_path:
            raise ConfigError("reverse-consistency 需要 --pool region_identify=<问法池文件>")
        derived = derive_reverse_consistency(
            records,
            load_predictions(args.preds),
            question_pool=PhrasePool.load(pool_path),
            scale=args.scale,
            tasks=tasks,
        )
    else:
        pools = {k: PhrasePool.load(v) for k, v in _key_value_pairs(args.pools, "--pool").items()}
        if not pools:
            raise ConfigError("question-perturbation 需要至少一个 --pool")
        derived = derive_question_perturbation(
            records,
            pools=pools,
            variants=args.variants,
            sample_ratio=args.sample_ratio,
            tasks=tasks,
        )

    if not derived:
        raise ConfigError(
            "派生集是空的。检查 --pred 是不是对应轮次的预测、--tasks 有没有拼错 —— "
            "写出一份空的评估集，下一步推理会正常跑完然后报表上多一格「样本不足」，"
            "而那格实际上是这里的配置错了。"
        )
    path = write_jsonl(derived, args.out)
    print(f"派生 {len(derived)} 条 -> {path}")
    return path


def _handle_check(args: argparse.Namespace) -> int:
    """跑之前先体检。整套跑完几个小时，而最常见的失败是一条路径写错。"""
    from .preflight import preflight, render

    result = preflight(_require_pipeline(args.config))
    print(render(result))
    return result.errors


HANDLERS: dict[str, Callable[[argparse.Namespace], Any]] = {
    "check": _handle_check,
    "convert": _handle_convert,
    "infer": _handle_infer,
    "eval": _handle_eval,
    "sweep": _handle_sweep,
    "all": _handle_all,
    "build-dpo": _handle_build_dpo,
    "derive": _handle_derive,
}


def main(argv: list[str] | None = None) -> Any:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["-h"], ["--help"]):
        return build_parser().parse_args(arguments)
    if arguments and arguments[0] not in SUBCOMMANDS and not arguments[0].startswith("-"):
        # 第一个参数不是选项、又不是已知子命令 —— 用户是想敲子命令但敲错了（或者
        # 代码是旧的、还没有这个子命令）。落到老通路的话，报的是老解析器的
        # "unrecognized arguments: check"，看起来像参数写错了，实际上是命令不存在。
        raise SystemExit(
            f"未知的子命令：{arguments[0]!r}\n"
            f"可用的：{', '.join(sorted(SUBCOMMANDS))}\n"
            "（这个子命令是新加的话，先 git pull 更新代码）"
        )
    if not arguments or arguments[0] not in SUBCOMMANDS:
        return legacy_eval_main(arguments)
    parser = build_parser()
    args = parser.parse_args(arguments)
    try:
        return HANDLERS[args.command](args)
    # DpoConfigError derives from ConfigError, so config and pipeline failures in
    # the DPO builder render through the same parser error path as every command.
    except (ConfigError, DpoPipelineError, PipelineError, RubricError) as exc:
        parser.error(str(exc))
