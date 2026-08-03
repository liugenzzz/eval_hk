from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import ArtifactLayout
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
        )


DEFAULT_DATASETS = {"mcq": "aero_mcq", "judge": "aero_judge", "vqa": "aero_vqa"}


def is_pipeline_config(path: str | Path) -> bool:
    raw = _load_raw_config(Path(path))
    infer_raw = raw.get("infer")
    models_raw = raw.get("models")
    return (
        isinstance(infer_raw, dict)
        and isinstance(models_raw, list)
        and any(
            isinstance(model, dict)
            and ("model_path" in model or "pred" in model)
            for model in models_raw
        )
    )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).resolve()
    raw = _load_raw_config(config_path)
    base_dir = config_path.parent

    datasets_raw = raw.get("datasets") or DEFAULT_DATASETS
    if not isinstance(datasets_raw, dict) or not datasets_raw:
        raise ConfigError("datasets must be a non-empty object")
    datasets = {str(key): str(value) for key, value in datasets_raw.items()}
    if any(not key.strip() or not value.strip() for key, value in datasets.items()):
        raise ConfigError("datasets keys and names must be non-empty")

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
    infer = PipelineInferSettings(
        prompt_files=prompt_files,
        max_new_tokens=int(infer_raw.get("max_new_tokens", 512)),
        batch_size=max(1, int(infer_raw.get("batch_size", 1))),
        limit=int(limit_raw) if limit_raw is not None else None,
        torch_dtype=str(infer_raw.get("torch_dtype", "auto")),
        device_map=str(infer_raw.get("device_map", "auto")),
        gpu_ids=[int(item) for item in infer_raw.get("gpu_ids", [])],
        workers_per_gpu=max(1, int(infer_raw.get("workers_per_gpu", 1))),
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
    datasets_raw = raw.get("datasets") or raw.get("DATASETS")
    datasets = {str(k): str(v) for k, v in datasets_raw.items()} if datasets_raw else dict(DEFAULT_DATASETS)
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
    )


def load_infer_config(path: str | Path) -> InferConfig:
    path = Path(path)
    raw = _load_raw_config(path)
    base_dir = path.parent
    infer_raw = raw.get("infer") or raw.get("INFER") or raw
    datasets_raw = infer_raw.get("datasets") or infer_raw.get("DATASETS")
    datasets = {str(k): str(v) for k, v in datasets_raw.items()} if datasets_raw else dict(DEFAULT_DATASETS)
    prompt_files_raw = infer_raw.get("prompt_files") or infer_raw.get("PROMPT_FILES") or {}
    prompt_files = {str(k): _resolve_path(v, base_dir) for k, v in prompt_files_raw.items()}
    limit_value = infer_raw.get("limit") or infer_raw.get("LIMIT")
    gpu_ids = infer_raw.get("gpu_ids") or infer_raw.get("GPU_IDS") or []
    if "overwrite" in infer_raw:
        overwrite = bool(infer_raw["overwrite"])
    elif "OVERWRITE" in infer_raw:
        overwrite = bool(infer_raw["OVERWRITE"])
    else:
        overwrite = None
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
        resume=bool(infer_raw.get("resume", infer_raw.get("RESUME", False))),
        clean_partial=bool(
            infer_raw.get("clean_partial", infer_raw.get("CLEAN_PARTIAL", False))
        ),
    )


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
