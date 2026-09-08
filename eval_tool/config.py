from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import ArtifactLayout
from .imaging import validate_image_pixel_bounds
from .judge import JudgeSettings
from .prompting import load_prompt_text


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelConfig:
    name: str
    paths: dict[str, str] = field(default_factory=dict)
    # Optional: dataset_key -> path to a previously written detail_<model>_<dataset>.xlsx
    # (or judge_detail_all.xlsx filtered to one model). When set, run_eval loads the
    # already-scored rows directly and skips re-reading predictions / re-calling the
    # judge for that model+dataset entirely -- for a baseline that never changes between
    # runs, this avoids paying the judge API cost again on every eval.
    scored_paths: dict[str, str] = field(default_factory=dict)

    def path_for(self, dataset_key: str) -> str | None:
        return self.paths.get(dataset_key)

    def scored_path_for(self, dataset_key: str) -> str | None:
        return self.scored_paths.get(dataset_key)


DEFAULT_CATEGORY_WEIGHTS = {"P1": 1.0, "P2": 1.0, "P3": 1.0, "R1": 1.0, "R2": 1.0, "R3": 1.0}


# 老配置的三个数据集键名和打分方式是一一对应的，那时 run_eval 直接按键名分支。
# 现在打分方式由 kind 决定，但这三个键名的历史含义要保住：不写 kind 的旧配置
# 必须还能跑。除这三个之外的任何键都必须显式声明 kind —— 猜错了会静默用错打分器，
# 那比报错难查得多。
DEFAULT_DATASET_KINDS = {"mcq": "choice", "judge": "choice", "vqa": "judge_text"}


@dataclass(frozen=True)
class EvalConfig:
    tsv_dir: Path
    out_dir: Path
    cache_dir: Path
    datasets: dict[str, str]
    models: list[ModelConfig]
    baseline_model: str = "base"
    judge: JudgeSettings = field(default_factory=JudgeSettings)
    max_workers: int = 8
    do_pointwise: bool = True
    do_pairwise: bool = True
    do_length_control: bool = True
    mcq_llm_extract_fallback: bool = False
    bootstrap_n: int = 1000
    seed: int = 42
    enabled_datasets: list[str] = field(default_factory=lambda: ["mcq", "judge", "vqa"])
    category_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CATEGORY_WEIGHTS))
    # dataset_key -> 打分器 kind / 打分器参数。留空则回落到 DEFAULT_DATASET_KINDS。
    dataset_kinds: dict[str, str] = field(default_factory=dict)
    dataset_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    # 配置文件所在目录。params 里的相对路径（类别表、词表、问法池）按它解析，
    # 这样一份配置在哪台机器上跑都指得对。
    base_dir: Path | None = None
    # 报表：拆分维度、显式声明的空格子、验收总分的数据集权重。
    report_dims: list[dict[str, Any]] = field(default_factory=list)
    empty_cells: dict[str, Any] = field(default_factory=dict)
    dataset_weights: dict[str, float] = field(default_factory=dict)
    # 链路衰减率的数据集配对：[{"gold": "describe", "model": "describe_modelhist"}]
    chain_decay_pairs: list[dict[str, str]] = field(default_factory=list)
    # 打分口径版本。阈值或判据改了就该动它 —— 它进指纹，口径不同的结果不许
    # 画进同一张对比图。
    profile_name: str = ""
    profile_version: str = ""

    def kind_for(self, dataset_key: str) -> str:
        kind = self.dataset_kinds.get(dataset_key) or DEFAULT_DATASET_KINDS.get(dataset_key)
        if not kind:
            raise ConfigError(
                f"datasets.{dataset_key} 没有声明 kind，也不在默认映射里。"
                f"写成 {{\"name\": ..., \"kind\": ...}}"
            )
        return kind

    def params_for(self, dataset_key: str) -> dict[str, Any]:
        return dict(self.dataset_params.get(dataset_key) or {})


@dataclass(frozen=True)
class InferConfig:
    model_name: str
    model_path: Path
    tsv_dir: Path
    out_dir: Path
    datasets: dict[str, str]
    prompt_files: dict[str, Path]
    max_new_tokens: int = 512
    batch_size: int = 1
    limit: int | None = None
    overwrite: bool | None = None
    torch_dtype: str = "auto"
    device_map: str = "auto"
    gpu_ids: list[int] = field(default_factory=list)
    workers_per_gpu: int = 1
    resume: bool = False
    clean_partial: bool = False
    image_min_pixels: int | None = None
    image_max_pixels: int | None = None


@dataclass(frozen=True)
class PipelineModelConfig:
    name: str
    model_path: Path | None
    pred_paths: dict[str, Path] = field(default_factory=dict)
    scored_paths: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineInferSettings:
    prompt_files: dict[str, Path] = field(default_factory=dict)
    max_new_tokens: int = 512
    batch_size: int = 1
    limit: int | None = None
    torch_dtype: str = "auto"
    device_map: str = "auto"
    gpu_ids: list[int] = field(default_factory=list)
    workers_per_gpu: int = 1
    image_min_pixels: int | None = None
    image_max_pixels: int | None = None


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    tsv_dir: Path
    work_dir: Path
    out_dir: Path
    cache_dir: Path
    datasets: dict[str, str]
    models: list[PipelineModelConfig]
    baseline_model: str
    infer: PipelineInferSettings
    judge: JudgeSettings
    convert_input: Path | None = None
    max_workers: int = 8
    do_pointwise: bool = True
    do_pairwise: bool = True
    do_length_control: bool = True
    mcq_llm_extract_fallback: bool = False
    bootstrap_n: int = 1000
    seed: int = 42
    enabled_datasets: list[str] = field(
        default_factory=lambda: ["mcq", "judge", "vqa"]
    )
    category_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_WEIGHTS)
    )
    dataset_kinds: dict[str, str] = field(default_factory=dict)
    dataset_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    report_dims: list[dict[str, Any]] = field(default_factory=list)
    empty_cells: dict[str, Any] = field(default_factory=dict)
    dataset_weights: dict[str, float] = field(default_factory=dict)
    chain_decay_pairs: list[dict[str, str]] = field(default_factory=list)
    profile_name: str = ""
    profile_version: str = ""

    @property
    def artifacts(self) -> ArtifactLayout:
        return ArtifactLayout(self.work_dir, self.out_dir)

    def select_models(
        self, names: list[str] | tuple[str, ...] | None = None
    ) -> list[PipelineModelConfig]:
        by_name = {model.name: model for model in self.models}
        requested = list(by_name) if names is None else [str(name) for name in names]
        unknown = [name for name in requested if name not in by_name]
        if unknown:
            raise ConfigError(f"unknown models: {','.join(unknown)}")
        if len(set(requested)) != len(requested):
            raise ConfigError("duplicate names in --models")
        return [by_name[name] for name in requested]

    def to_infer_configs(
        self,
        names: list[str] | tuple[str, ...] | None = None,
        *,
        overwrite: bool = False,
        clean_partial: bool = False,
    ) -> list[InferConfig]:
        configs: list[InferConfig] = []
        for model in self.select_models(names):
            pending_datasets = {
                key: self.datasets[key]
                for key in self.enabled_datasets
                if key not in model.scored_paths and key not in model.pred_paths
            }
            if not pending_datasets:
                continue
            if model.model_path is None:
                raise ConfigError(
                    f"model_path is required for inference model: {model.name}"
                )
            configs.append(
                InferConfig(
                    model_name=model.name,
                    model_path=model.model_path,
                    tsv_dir=self.tsv_dir,
                    out_dir=self.artifacts.model_dir(model.name),
                    datasets=pending_datasets,
                    prompt_files=dict(self.infer.prompt_files),
                    max_new_tokens=self.infer.max_new_tokens,
                    batch_size=self.infer.batch_size,
                    limit=self.infer.limit,
                    overwrite=overwrite,
                    torch_dtype=self.infer.torch_dtype,
                    device_map=self.infer.device_map,
                    gpu_ids=list(self.infer.gpu_ids),
                    workers_per_gpu=self.infer.workers_per_gpu,
                    resume=True,
                    clean_partial=clean_partial,
                    image_min_pixels=self.infer.image_min_pixels,
                    image_max_pixels=self.infer.image_max_pixels,
                )
            )
        return configs

    def to_eval_config(
        self, names: list[str] | tuple[str, ...] | None = None
    ) -> EvalConfig:
        selected = self.select_models(names)
        models: list[ModelConfig] = []
        for model in selected:
            paths: dict[str, str] = {}
            scored_paths: dict[str, str] = {}
            for dataset_key in self.enabled_datasets:
                if dataset_key in model.scored_paths:
                    scored_paths[dataset_key] = str(model.scored_paths[dataset_key])
                elif dataset_key in model.pred_paths:
                    paths[dataset_key] = str(model.pred_paths[dataset_key])
                else:
                    paths[dataset_key] = str(
                        self.artifacts.prediction(
                            model.name, self.datasets[dataset_key]
                        )
                    )
            models.append(
                ModelConfig(
                    name=model.name,
                    paths=paths,
                    scored_paths=scored_paths,
                )
            )
        selected_names = {model.name for model in selected}
        return EvalConfig(
            tsv_dir=self.tsv_dir,
            out_dir=self.out_dir,
            cache_dir=self.cache_dir,
            datasets=dict(self.datasets),
            models=models,
            baseline_model=self.baseline_model,
            judge=self.judge,
            max_workers=self.max_workers,
            do_pointwise=self.do_pointwise,
            do_pairwise=self.do_pairwise
            and self.baseline_model in selected_names,
            do_length_control=self.do_length_control,
            mcq_llm_extract_fallback=self.mcq_llm_extract_fallback,
            bootstrap_n=self.bootstrap_n,
            seed=self.seed,
            enabled_datasets=list(self.enabled_datasets),
            category_weights=dict(self.category_weights),
            dataset_kinds=dict(self.dataset_kinds),
            dataset_params={k: dict(v) for k, v in self.dataset_params.items()},
            base_dir=self.config_path.parent,
            report_dims=[dict(d) for d in self.report_dims],
            empty_cells=dict(self.empty_cells),
            dataset_weights=dict(self.dataset_weights),
            chain_decay_pairs=[dict(p) for p in self.chain_decay_pairs],
            profile_name=self.profile_name,
            profile_version=self.profile_version,
        )


DEFAULT_DATASETS = {"mcq": "aero_mcq", "judge": "aero_judge", "vqa": "aero_vqa"}


def parse_datasets(
    datasets_raw: Any, *, where: str = "datasets"
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, Any]]]:
    """解析 datasets 块，返回 (名称表, kind 表, 参数表)。

    两种写法等价，字符串是旧写法：

        "datasets": {
          "mcq": "aero_mcq",
          "ground": {"name": "eval_set_v1", "kind": "grounding_single",
                     "params": {"iou_gate": 0.5}}
        }

    这里只校验形状，不校验 kind 是否已实现 —— 那是 run_eval 起跑前一次性查注册表
    的事，配置层不该 import 打分器（会绕成循环导入，且 --help 也得付代价）。
    """
    if not isinstance(datasets_raw, dict) or not datasets_raw:
        raise ConfigError(f"{where} must be a non-empty object")
    names: dict[str, str] = {}
    kinds: dict[str, str] = {}
    params: dict[str, dict[str, Any]] = {}
    for key_raw, value in datasets_raw.items():
        key = str(key_raw).strip()
        if not key:
            raise ConfigError(f"{where} keys must be non-empty")
        if isinstance(value, dict):
            name = str(value.get("name", "")).strip()
            kind = str(value.get("kind", "")).strip()
            params_raw = value.get("params", {})
            if params_raw is None:
                params_raw = {}
            if not isinstance(params_raw, dict):
                raise ConfigError(f"{where}.{key}.params must be an object")
            if kind:
                kinds[key] = kind
            if params_raw:
                params[key] = dict(params_raw)
        else:
            name = str(value).strip()
        if not name:
            raise ConfigError(f"{where}.{key} must name a dataset")
        names[key] = name
    return names, kinds, params


def is_pipeline_config(path: str | Path) -> bool:
    raw = _load_raw_config(Path(path))
    infer_raw = raw.get("infer")
    models_raw = raw.get("models")
    return (
        isinstance(infer_raw, dict)
        and isinstance(models_raw, list)
        and any(
            isinstance(model, dict)
            and (
                "model_path" in model
                or "pred" in model
                or "scored" in model
            )
            for model in models_raw
        )
    )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    raw = _load_raw_config(config_path)
    base_dir = config_path.parent

    datasets, dataset_kinds, dataset_params = parse_datasets(
        raw.get("datasets") or DEFAULT_DATASETS
    )
    report_dims, empty_cells, dataset_weights, chain_decay_pairs = _parse_report_block(raw)
    profile_name, profile_version = _parse_profile_block(raw)

    enabled_raw = raw.get("enabled_datasets", list(datasets))
    if not isinstance(enabled_raw, list):
        raise ConfigError("enabled_datasets must be a list")
    enabled_datasets = [str(item) for item in enabled_raw]
    unknown_enabled = [key for key in enabled_datasets if key not in datasets]
    if unknown_enabled:
        raise ConfigError(
            f"unknown enabled datasets: {','.join(unknown_enabled)}"
        )

    models_raw = raw.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise ConfigError("models must be a non-empty list")
    models: list[PipelineModelConfig] = []
    seen_names: set[str] = set()
    for position, item in enumerate(models_raw):
        if not isinstance(item, dict):
            raise ConfigError(f"models[{position}] must be an object")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ConfigError(f"models[{position}].name must be non-empty")
        if name in seen_names:
            raise ConfigError(f"duplicate model name: {name}")
        seen_names.add(name)
        pred_paths = _pipeline_path_map(
            item.get("pred"), base_dir, datasets, f"models[{position}].pred"
        )
        scored_paths = _pipeline_path_map(
            item.get("scored"), base_dir, datasets, f"models[{position}].scored"
        )
        model_path_raw = item.get("model_path")
        model_path = (
            _resolve_path(model_path_raw, base_dir) if model_path_raw else None
        )
        missing = [
            key
            for key in enabled_datasets
            if key not in pred_paths and key not in scored_paths
        ]
        if model_path is None and missing:
            raise ConfigError(
                f"models[{position}].model_path is required for datasets: "
                f"{','.join(missing)}"
            )
        models.append(
            PipelineModelConfig(
                name=name,
                model_path=model_path,
                pred_paths=pred_paths,
                scored_paths=scored_paths,
            )
        )

    baseline_model = str(raw.get("baseline_model") or "base")
    if baseline_model not in seen_names:
        raise ConfigError(f"baseline_model not found in models: {baseline_model}")

    infer_raw = raw.get("infer")
    if not isinstance(infer_raw, dict):
        raise ConfigError("infer must be an object")
    prompt_files = _pipeline_path_map(
        infer_raw.get("prompt_files"),
        base_dir,
        datasets,
        "infer.prompt_files",
    )
    limit_raw = infer_raw.get("limit")
    image_min_pixels, image_max_pixels = _config_image_pixel_bounds(infer_raw)
    _assert_training_pixel_budget(infer_raw, image_min_pixels, image_max_pixels)
    infer = PipelineInferSettings(
        prompt_files=prompt_files,
        max_new_tokens=int(infer_raw.get("max_new_tokens", 512)),
        batch_size=max(1, int(infer_raw.get("batch_size", 1))),
        limit=int(limit_raw) if limit_raw is not None else None,
        torch_dtype=str(infer_raw.get("torch_dtype", "auto")),
        device_map=str(infer_raw.get("device_map", "auto")),
        gpu_ids=[int(item) for item in infer_raw.get("gpu_ids", [])],
        workers_per_gpu=max(1, int(infer_raw.get("workers_per_gpu", 1))),
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )

    judge = _pipeline_judge_settings(raw.get("judge"), base_dir)
    convert_raw = raw.get("convert") or {}
    if not isinstance(convert_raw, dict):
        raise ConfigError("convert must be an object")
    convert_value = convert_raw.get("input_json")
    convert_input = _resolve_path(convert_value, base_dir) if convert_value else None

    category_weights = dict(DEFAULT_CATEGORY_WEIGHTS)
    weights_raw = raw.get("category_weights") or {}
    if not isinstance(weights_raw, dict):
        raise ConfigError("category_weights must be an object")
    category_weights.update(
        {str(key): float(value) for key, value in weights_raw.items()}
    )

    return PipelineConfig(
        config_path=config_path,
        tsv_dir=_resolve_path(raw.get("tsv_dir") or ".", base_dir),
        work_dir=_resolve_path(raw.get("work_dir") or "work_dir", base_dir),
        out_dir=_resolve_path(raw.get("out_dir") or "eval_report", base_dir),
        cache_dir=_resolve_path(raw.get("cache_dir") or "eval_cache", base_dir),
        datasets=datasets,
        dataset_kinds=dataset_kinds,
        dataset_params=dataset_params,
        report_dims=report_dims,
        empty_cells=empty_cells,
        dataset_weights=dataset_weights,
        chain_decay_pairs=chain_decay_pairs,
        profile_name=profile_name,
        profile_version=profile_version,
        models=models,
        baseline_model=baseline_model,
        infer=infer,
        judge=judge,
        convert_input=convert_input,
        max_workers=int(raw.get("max_workers", 8)),
        do_pointwise=bool(raw.get("do_pointwise", True)),
        do_pairwise=bool(raw.get("do_pairwise", True)),
        do_length_control=bool(raw.get("do_length_control", True)),
        mcq_llm_extract_fallback=bool(
            raw.get("mcq_llm_extract_fallback", False)
        ),
        bootstrap_n=int(raw.get("bootstrap_n", 1000)),
        seed=int(raw.get("seed", 42)),
        enabled_datasets=enabled_datasets,
        category_weights=category_weights,
    )


def load_config(path: str | Path) -> EvalConfig:
    path = Path(path)
    raw = _load_raw_config(path)
    base_dir = path.parent
    datasets_raw = raw.get("datasets") or raw.get("DATASETS") or DEFAULT_DATASETS
    datasets, dataset_kinds, dataset_params = parse_datasets(datasets_raw)
    report_dims, empty_cells, dataset_weights, chain_decay_pairs = _parse_report_block(raw)
    profile_name, profile_version = _parse_profile_block(raw)
    models_raw = raw.get("models") or raw.get("MODELS") or []
    models = [
        ModelConfig(
            name=str(m["name"]),
            paths={k: str(v) for k, v in m.items() if k not in ("name", "scored")},
            scored_paths={k: str(v) for k, v in (m.get("scored") or {}).items()},
        )
        for m in models_raw
    ]
    enabled_datasets = [str(x) for x in (raw.get("enabled_datasets") or raw.get("ENABLED_DATASETS") or list(datasets.keys()))]
    category_weights = dict(DEFAULT_CATEGORY_WEIGHTS)
    category_weights.update({str(k): float(v) for k, v in (raw.get("category_weights") or raw.get("CATEGORY_WEIGHTS") or {}).items()})
    judge_raw = raw.get("judge") or {}
    judge_prompt_files = judge_raw.get("prompt_files") or raw.get("judge_prompt_files") or {}
    pointwise_prompt = _optional_prompt(
        judge_prompt_files.get("pointwise") or judge_prompt_files.get("vqa_pointwise"),
        base_dir,
        JudgeSettings.pointwise_prompt,
    )
    pairwise_prompt = _optional_prompt(
        judge_prompt_files.get("pairwise") or judge_prompt_files.get("vqa_pairwise"),
        base_dir,
        JudgeSettings.pairwise_prompt,
    )
    judge = JudgeSettings(
        api_base=str(judge_raw.get("api_base") or raw.get("JUDGE_API_BASE") or JudgeSettings.api_base),
        api_key=str(judge_raw.get("api_key") or raw.get("JUDGE_API_KEY") or JudgeSettings.api_key),
        model=str(judge_raw.get("model") or raw.get("JUDGE_MODEL") or JudgeSettings.model),
        temperature=float(judge_raw.get("temperature") or raw.get("JUDGE_TEMP") or JudgeSettings.temperature),
        timeout=int(judge_raw.get("timeout") or raw.get("TIMEOUT") or JudgeSettings.timeout),
        max_retries=int(judge_raw.get("max_retries") or raw.get("MAX_RETRIES") or JudgeSettings.max_retries),
        pointwise_prompt=pointwise_prompt,
        pairwise_prompt=pairwise_prompt,
    )
    return EvalConfig(
        tsv_dir=_resolve_path(raw.get("tsv_dir") or raw.get("TSV_DIR") or ".", base_dir),
        out_dir=_resolve_path(raw.get("out_dir") or raw.get("OUT_DIR") or "eval_report", base_dir),
        cache_dir=_resolve_path(raw.get("cache_dir") or raw.get("CACHE_DIR") or "eval_cache", base_dir),
        datasets=datasets,
        models=models,
        baseline_model=str(raw.get("baseline_model") or raw.get("BASELINE_MODEL") or "base"),
        judge=judge,
        max_workers=int(raw.get("max_workers") or raw.get("MAX_WORKERS") or 8),
        do_pointwise=bool(raw.get("do_pointwise", raw.get("DO_POINTWISE", True))),
        do_pairwise=bool(raw.get("do_pairwise", raw.get("DO_PAIRWISE", True))),
        do_length_control=bool(raw.get("do_length_control", raw.get("DO_LENGTH_CONTROL", True))),
        mcq_llm_extract_fallback=bool(raw.get("mcq_llm_extract_fallback", raw.get("MCQ_LLM_EXTRACT_FALLBACK", False))),
        bootstrap_n=int(raw.get("bootstrap_n") or raw.get("BOOTSTRAP_N") or 1000),
        seed=int(raw.get("seed") or raw.get("SEED") or 42),
        enabled_datasets=enabled_datasets,
        category_weights=category_weights,
        dataset_kinds=dataset_kinds,
        dataset_params=dataset_params,
        base_dir=base_dir,
        report_dims=report_dims,
        empty_cells=empty_cells,
        dataset_weights=dataset_weights,
        chain_decay_pairs=chain_decay_pairs,
        profile_name=profile_name,
        profile_version=profile_version,
    )


def load_infer_config(path: str | Path) -> InferConfig:
    path = Path(path)
    raw = _load_raw_config(path)
    base_dir = path.parent
    infer_raw = raw.get("infer") or raw.get("INFER") or raw
    datasets_raw = infer_raw.get("datasets") or infer_raw.get("DATASETS") or DEFAULT_DATASETS
    # 推理不关心 kind，但配置文件是同一份，得认得对象写法。
    datasets, _, _ = parse_datasets(datasets_raw, where="infer.datasets")
    prompt_files_raw = infer_raw.get("prompt_files") or infer_raw.get("PROMPT_FILES") or {}
    prompt_files = {str(k): _resolve_path(v, base_dir) for k, v in prompt_files_raw.items()}
    limit_value = infer_raw.get("limit") or infer_raw.get("LIMIT")
    gpu_ids = infer_raw.get("gpu_ids") or infer_raw.get("GPU_IDS") or []
    overwrite = _infer_bool(infer_raw, "overwrite", "OVERWRITE", None)
    image_min_pixels, image_max_pixels = _config_image_pixel_bounds(
        infer_raw,
        allow_legacy_uppercase=True,
    )
    _assert_training_pixel_budget(infer_raw, image_min_pixels, image_max_pixels)
    return InferConfig(
        model_name=str(infer_raw.get("model_name") or infer_raw.get("MODEL_NAME")),
        model_path=_resolve_path(infer_raw.get("model_path") or infer_raw.get("MODEL_PATH"), base_dir),
        tsv_dir=_resolve_path(infer_raw.get("tsv_dir") or infer_raw.get("TSV_DIR") or ".", base_dir),
        out_dir=_resolve_path(infer_raw.get("out_dir") or infer_raw.get("OUT_DIR") or "work_dir", base_dir),
        datasets=datasets,
        prompt_files=prompt_files,
        max_new_tokens=int(infer_raw.get("max_new_tokens") or infer_raw.get("MAX_NEW_TOKENS") or 512),
        batch_size=max(1, int(infer_raw.get("batch_size") or infer_raw.get("BATCH_SIZE") or 1)),
        limit=int(limit_value) if limit_value else None,
        overwrite=overwrite,
        torch_dtype=str(infer_raw.get("torch_dtype") or infer_raw.get("TORCH_DTYPE") or "auto"),
        device_map=str(infer_raw.get("device_map") or infer_raw.get("DEVICE_MAP") or "auto"),
        gpu_ids=[int(x) for x in gpu_ids],
        workers_per_gpu=max(1, int(infer_raw.get("workers_per_gpu") or infer_raw.get("WORKERS_PER_GPU") or 1)),
        resume=_infer_bool(infer_raw, "resume", "RESUME", False),
        clean_partial=_infer_bool(
            infer_raw, "clean_partial", "CLEAN_PARTIAL", False
        ),
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
    )


def _assert_training_pixel_budget(
    raw: dict[str, Any], image_min_pixels: int | None, image_max_pixels: int | None
) -> None:
    """§18.4 像素预算硬校验：推理的像素面积必须与训练时的 LLaMAFactory 配置一致。

    分辨率变了，坐标虽是归一化的不会错位，但模型的空间精度会变，测出来的数字不可比 ——
    而这件事在报表上完全看不出来，只会表现为「这个 checkpoint 好像差一点」。

        "infer": {
          "image_min_pixels": 65536, "image_max_pixels": 589824,
          "training": {"image_min_pixels": 65536, "image_max_pixels": 589824}
        }

    写了 training 就硬校验，不一致直接报错。训练配置会变，所以期望值从配置来，
    不在代码里写死一组常量。
    """
    training = raw.get("training")
    if training is None:
        return
    if not isinstance(training, dict):
        raise ConfigError("infer.training must be an object")
    expected_min = training.get("image_min_pixels")
    expected_max = training.get("image_max_pixels")
    if expected_min is None or expected_max is None:
        raise ConfigError(
            "infer.training 要同时给 image_min_pixels 和 image_max_pixels"
        )
    if image_min_pixels is None or image_max_pixels is None:
        raise ConfigError(
            "infer.training 声明了训练时的像素面积，但推理没设 image_min_pixels /"
            " image_max_pixels：那等于按旧行为不做预缩放，和训练时不一致"
        )
    if (image_min_pixels, image_max_pixels) != (int(expected_min), int(expected_max)):
        raise ConfigError(
            f"推理像素面积 ({image_min_pixels}, {image_max_pixels}) 与训练配置 "
            f"({int(expected_min)}, {int(expected_max)}) 不一致。分辨率变了模型的空间精度"
            f"就变了，测出来的数字不可比。"
        )


def _config_image_pixel_bounds(
    raw: dict[str, Any],
    *,
    allow_legacy_uppercase: bool = False,
) -> tuple[int | None, int | None]:
    def select(name: str, legacy_name: str) -> tuple[bool, Any]:
        if name in raw:
            return True, raw[name]
        if allow_legacy_uppercase and legacy_name in raw:
            return True, raw[legacy_name]
        return False, None

    min_present, image_min_pixels = select(
        "image_min_pixels", "IMAGE_MIN_PIXELS"
    )
    max_present, image_max_pixels = select(
        "image_max_pixels", "IMAGE_MAX_PIXELS"
    )
    if min_present != max_present:
        raise ConfigError(
            "image_min_pixels and image_max_pixels must be provided together "
            "or both be omitted"
        )
    try:
        return validate_image_pixel_bounds(image_min_pixels, image_max_pixels)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def _infer_bool(
    raw: dict[str, Any],
    name: str,
    legacy_name: str,
    default: bool | None,
) -> bool | None:
    if name in raw:
        value = raw[name]
    elif legacy_name in raw:
        value = raw[legacy_name]
    else:
        return default
    if type(value) is bool:
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
    if type(value) is int and value in {0, 1}:
        return bool(value)
    raise ConfigError(f"{name} must be a boolean")


def _load_raw_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".py":
        spec = importlib.util.spec_from_file_location("eval_tool_user_config", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load config: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return {name: getattr(module, name) for name in dir(module) if not name.startswith("__")}
    raise ValueError(f"Unsupported config format: {path}")


def _resolve_path(value: object, base_dir: Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (base_dir / path).resolve()


def _optional_prompt(value: object, base_dir: Path, default: str) -> str:
    if not value:
        return default
    return load_prompt_text(_resolve_path(value, base_dir))


def _pipeline_path_map(
    value: object,
    base_dir: Path,
    datasets: dict[str, str],
    field_name: str,
) -> dict[str, Path]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{field_name} must be an object")
    unknown = [str(key) for key in value if str(key) not in datasets]
    if unknown:
        raise ConfigError(
            f"unknown dataset in {field_name}: {','.join(unknown)}"
        )
    resolved: dict[str, Path] = {}
    for key, path_value in value.items():
        if not str(path_value).strip():
            raise ConfigError(f"{field_name}.{key} must be a non-empty path")
        resolved[str(key)] = _resolve_path(path_value, base_dir)
    return resolved


def _pipeline_judge_settings(value: object, base_dir: Path) -> JudgeSettings:
    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, dict):
        raw = value
    else:
        raise ConfigError("judge must be an object")
    prompt_files = raw.get("prompt_files") or {}
    if not isinstance(prompt_files, dict):
        raise ConfigError("judge.prompt_files must be an object")
    unknown_prompts = [
        str(key) for key in prompt_files if str(key) not in {"pointwise", "pairwise"}
    ]
    if unknown_prompts:
        raise ConfigError(
            f"unknown judge prompt files: {','.join(unknown_prompts)}"
        )
    return JudgeSettings(
        api_base=str(raw.get("api_base", JudgeSettings.api_base)),
        api_key=str(raw.get("api_key", JudgeSettings.api_key)),
        model=str(raw.get("model", JudgeSettings.model)),
        temperature=float(raw.get("temperature", JudgeSettings.temperature)),
        timeout=int(raw.get("timeout", JudgeSettings.timeout)),
        max_retries=int(raw.get("max_retries", JudgeSettings.max_retries)),
        pointwise_prompt=_optional_prompt(
            prompt_files.get("pointwise"),
            base_dir,
            JudgeSettings.pointwise_prompt,
        ),
        pairwise_prompt=_optional_prompt(
            prompt_files.get("pairwise"),
            base_dir,
            JudgeSettings.pairwise_prompt,
        ),
    )


def _parse_report_block(
    raw: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float], list[dict[str, str]]]:
    """report 块：拆分维度、显式声明的空格子、验收总分的数据集权重。

        "report": {
          "dims": [{"key": "size_bucket", "from": "meta.size_bucket"}],
          "empty_cells": {"by_design": [["ground_part", "hard"]],
                          "not_in_data": ["ground_unique"]},
          "dataset_weights": {"ground_box": 0.40, "region_identify": 0.25}
        }
    """
    report_raw = raw.get("report") or {}
    if not isinstance(report_raw, dict):
        raise ConfigError("report must be an object")
    dims_raw = report_raw.get("dims") or []
    if not isinstance(dims_raw, list):
        raise ConfigError("report.dims must be a list")
    empty_raw = report_raw.get("empty_cells") or {}
    if not isinstance(empty_raw, dict):
        raise ConfigError("report.empty_cells must be an object")
    weights_raw = report_raw.get("dataset_weights") or {}
    if not isinstance(weights_raw, dict):
        raise ConfigError("report.dataset_weights must be an object")
    decay_raw = report_raw.get("chain_decay") or []
    if not isinstance(decay_raw, list):
        raise ConfigError("report.chain_decay must be a list")
    for entry in decay_raw:
        if not isinstance(entry, dict) or not entry.get("gold") or not entry.get("model"):
            raise ConfigError('report.chain_decay entries need {"gold": ..., "model": ...}')
    return (
        [dict(item) for item in dims_raw],
        dict(empty_raw),
        {str(k): float(v) for k, v in weights_raw.items()},
        [{"gold": str(e["gold"]), "model": str(e["model"])} for e in decay_raw],
    )


def _parse_profile_block(raw: dict[str, Any]) -> tuple[str, str]:
    """profile 块只回答一个问题：这种 kind 的数据，用哪个打分器、哪版提示词。

        "profile": {"name": "grounding_zh_v1", "version": "v1"}

    具体的绑定写在 datasets 的 kind / params 里，这里只留名字和版本 —— 版本进指纹，
    口径改了而版本没动，两次评估就会被当成可比的，那是最难查的一类错。
    """
    profile_raw = raw.get("profile") or {}
    if not isinstance(profile_raw, dict):
        raise ConfigError("profile must be an object")
    name = str(profile_raw.get("name") or "")
    version = str(profile_raw.get("version") or name)
    return name, version
