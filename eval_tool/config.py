from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .judge import JudgeSettings
from .prompting import load_prompt_text


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


DEFAULT_DATASETS = {"mcq": "aero_mcq", "judge": "aero_judge", "vqa": "aero_vqa"}


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
