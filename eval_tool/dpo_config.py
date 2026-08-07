from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Literal, Sequence

from .config import ConfigError
from .judge import JudgeSettings


RubricName = Literal["binary", "v4"]


class DpoConfigError(ConfigError):
    """Invalid configuration for the independent DPO builder."""


@dataclass(frozen=True)
class DpoInferConfig:
    model_name: str
    model_path: Path
    enable_thinking: bool = False
    max_new_tokens: int = 1024
    batch_size: int = 1
    torch_dtype: str = "auto"
    device_map: str = "auto"
    gpu_ids: tuple[int, ...] = ()
    workers_per_gpu: int = 1


@dataclass(frozen=True)
class DpoJudgeConfig:
    settings: JudgeSettings
    max_workers: int = 8


@dataclass(frozen=True)
class DpoBuildConfig:
    config_path: Path
    inputs: tuple[Path, ...]
    output_dir: Path
    output_name: str
    work_dir: Path
    image_root: Path
    wrong_only: bool
    rubric: RubricName | None
    infer: DpoInferConfig
    judge: DpoJudgeConfig | None


_TOP_LEVEL_KEYS = {
    "inputs",
    "output_dir",
    "output_name",
    "work_dir",
    "image_root",
    "wrong_only",
    "rubric",
    "infer",
    "judge",
}
_INFER_KEYS = {
    "model_name",
    "model_path",
    "enable_thinking",
    "max_new_tokens",
    "batch_size",
    "torch_dtype",
    "device_map",
    "gpu_ids",
    "workers_per_gpu",
}
_JUDGE_KEYS = {
    "api_base",
    "api_key",
    "model",
    "temperature",
    "timeout",
    "max_retries",
    "max_workers",
}
_REQUIRED_JUDGE_KEYS = {
    "api_base",
    "api_key",
    "model",
    "temperature",
    "timeout",
    "max_retries",
}
_RESERVED_OUTPUT_NAMES = {
    "audit_records.jsonl",
    "rejected_records.jsonl",
    "summary.json",
    "manifest.json",
    "warnings.log",
}
_WINDOWS_DEVICE_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    "clock$",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def load_dpo_config(
    path: str | Path,
    *,
    input_overrides: Sequence[str | Path] | None = None,
    invocation_dir: Path | None = None,
) -> DpoBuildConfig:
    """Load and validate the independent JSON DPO-builder configuration."""

    config_path = Path(path).resolve()
    invocation_base = (
        Path.cwd().resolve()
        if invocation_dir is None
        else Path(invocation_dir).resolve()
    )
    if not invocation_base.is_dir():
        raise DpoConfigError("invocation_dir must be an existing directory")

    raw = _load_json_object(config_path)
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, "DPO top-level")

    if "wrong_only" not in raw:
        raise DpoConfigError("wrong_only is required")
    wrong_only = _strict_bool(raw["wrong_only"], "wrong_only")

    base_dir = config_path.parent
    inputs = _load_inputs(
        raw.get("inputs"),
        base_dir=base_dir,
        input_overrides=input_overrides,
        invocation_dir=invocation_base,
    )

    output_dir = _config_path(raw.get("output_dir"), base_dir, "output_dir")
    work_dir = _config_path(raw.get("work_dir"), base_dir, "work_dir")
    _require_directory_if_present(output_dir, "output_dir")
    _require_directory_if_present(work_dir, "work_dir")

    output_name = _safe_output_name(raw.get("output_name"))
    if (output_dir / output_name).resolve().parent != output_dir.resolve():
        raise DpoConfigError("output_name must remain inside output_dir")

    if "image_root" in raw:
        image_root = _config_path(raw["image_root"], base_dir, "image_root")
        if not image_root.is_dir():
            raise DpoConfigError("image_root must be an existing directory")
    else:
        image_root = invocation_base

    infer = _load_infer(raw.get("infer"), base_dir)
    rubric, judge = _load_judge_mode(raw, wrong_only)

    return DpoBuildConfig(
        config_path=config_path,
        inputs=inputs,
        output_dir=output_dir,
        output_name=output_name,
        work_dir=work_dir,
        image_root=image_root,
        wrong_only=wrong_only,
        rubric=rubric,
        infer=infer,
        judge=judge,
    )


def _load_json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise DpoConfigError(f"DPO config file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DpoConfigError(f"cannot read DPO config: {exc}") from exc
    if not isinstance(value, dict):
        raise DpoConfigError("DPO config must be a JSON object")
    return value


def _reject_unknown_keys(
    value: dict[str, object], allowed: set[str], location: str
) -> None:
    unknown = sorted(str(key) for key in value if key not in allowed)
    if unknown:
        raise DpoConfigError(
            f"unknown {location} keys: {', '.join(unknown)}"
        )


def _strict_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise DpoConfigError(f"{field_name} must be a JSON boolean")
    return value


def _nonempty_string(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip():
        raise DpoConfigError(f"{field_name} must be a non-empty string")
    return value.strip()


def _config_path(value: object, base_dir: Path, field_name: str) -> Path:
    text = _nonempty_string(value, field_name)
    candidate = Path(text)
    return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()


def _override_path(value: object, base_dir: Path, field_name: str) -> Path:
    if isinstance(value, Path):
        text = str(value)
        if not text.strip():
            raise DpoConfigError(f"{field_name} must be a non-empty path")
    else:
        text = _nonempty_string(value, field_name)
    candidate = Path(text)
    return candidate.resolve() if candidate.is_absolute() else (base_dir / candidate).resolve()


def _load_inputs(
    configured: object,
    *,
    base_dir: Path,
    input_overrides: Sequence[str | Path] | None,
    invocation_dir: Path,
) -> tuple[Path, ...]:
    if input_overrides is None:
        if not isinstance(configured, list) or not configured:
            raise DpoConfigError("inputs must be a non-empty list")
        values: Sequence[object] = configured
        path_base = base_dir
        resolver = _config_path
    else:
        if isinstance(input_overrides, (str, bytes, Path)) or not input_overrides:
            raise DpoConfigError("input overrides must be a non-empty sequence")
        values = input_overrides
        path_base = invocation_dir
        resolver = _override_path

    resolved: list[Path] = []
    for index, value in enumerate(values):
        input_path = resolver(value, path_base, f"inputs[{index}]")
        if not input_path.is_file():
            raise DpoConfigError(f"inputs[{index}] must be an existing file: {input_path}")
        resolved.append(input_path)
    return tuple(resolved)


def _require_directory_if_present(path: Path, field_name: str) -> None:
    if path.exists() and not path.is_dir():
        raise DpoConfigError(f"{field_name} must be a directory path")


def _load_infer(value: object, base_dir: Path) -> DpoInferConfig:
    if not isinstance(value, dict):
        raise DpoConfigError("infer must be an object")
    _reject_unknown_keys(value, _INFER_KEYS, "infer")

    model_name = _nonempty_string(value.get("model_name"), "infer.model_name")
    model_path = _config_path(value.get("model_path"), base_dir, "infer.model_path")
    if not model_path.exists():
        raise DpoConfigError(f"infer.model_path does not exist: {model_path}")

    enable_thinking = _strict_bool(
        value.get("enable_thinking", False), "infer.enable_thinking"
    )
    max_new_tokens = _integer_at_least(
        value.get("max_new_tokens", 1024), 1, "infer.max_new_tokens"
    )
    batch_size = _integer_at_least(
        value.get("batch_size", 1), 1, "infer.batch_size"
    )
    workers_per_gpu = _integer_at_least(
        value.get("workers_per_gpu", 1), 1, "infer.workers_per_gpu"
    )
    torch_dtype = _nonempty_string(
        value.get("torch_dtype", "auto"), "infer.torch_dtype"
    )
    device_map = _nonempty_string(
        value.get("device_map", "auto"), "infer.device_map"
    )
    gpu_ids = _gpu_ids(value.get("gpu_ids", []))
    if gpu_ids:
        physical_numbers = re.findall(r"cuda\s*:\s*(\d+)", device_map.casefold())
        if any(number != "0" for number in physical_numbers):
            raise DpoConfigError(
                "infer.device_map must use auto or worker-local cuda:0 when gpu_ids are set"
            )

    return DpoInferConfig(
        model_name=model_name,
        model_path=model_path,
        enable_thinking=enable_thinking,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
        device_map=device_map,
        gpu_ids=gpu_ids,
        workers_per_gpu=workers_per_gpu,
    )


def _integer_at_least(value: object, minimum: int, field_name: str) -> int:
    if type(value) is not int or value < minimum:
        raise DpoConfigError(
            f"{field_name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _gpu_ids(value: object) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise DpoConfigError("infer.gpu_ids must be a list")
    gpu_ids: list[int] = []
    for index, gpu_id in enumerate(value):
        if type(gpu_id) is not int or gpu_id < 0:
            raise DpoConfigError(
                f"infer.gpu_ids[{index}] must be a non-negative integer"
            )
        gpu_ids.append(gpu_id)
    if len(gpu_ids) != len(set(gpu_ids)):
        raise DpoConfigError("infer.gpu_ids must not contain duplicates")
    return tuple(gpu_ids)


def _load_judge_mode(
    raw: dict[str, object], wrong_only: bool
) -> tuple[RubricName | None, DpoJudgeConfig | None]:
    judge_value = raw.get("judge")
    if isinstance(judge_value, dict):
        _reject_unknown_keys(judge_value, _JUDGE_KEYS, "judge")

    if not wrong_only:
        return None, None

    rubric_value = raw.get("rubric")
    if type(rubric_value) is not str or rubric_value not in {"binary", "v4"}:
        raise DpoConfigError("rubric is required and must be one of: binary, v4")
    rubric: RubricName = rubric_value

    if not isinstance(judge_value, dict):
        raise DpoConfigError("judge must be a complete object in wrong-only mode")
    missing = sorted(_REQUIRED_JUDGE_KEYS - judge_value.keys())
    if missing:
        raise DpoConfigError(f"judge.{missing[0]} is required")

    api_base = _nonempty_string(judge_value["api_base"], "judge.api_base")
    api_key = _nonempty_string(judge_value["api_key"], "judge.api_key")
    model = _nonempty_string(judge_value["model"], "judge.model")
    temperature = _temperature(judge_value["temperature"])
    timeout = _integer_at_least(judge_value["timeout"], 1, "judge.timeout")
    max_retries = _integer_at_least(
        judge_value["max_retries"], 0, "judge.max_retries"
    )
    max_workers = _integer_at_least(
        judge_value.get("max_workers", 8), 1, "judge.max_workers"
    )

    settings = JudgeSettings(
        api_base=api_base,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )
    return rubric, DpoJudgeConfig(settings=settings, max_workers=max_workers)


def _temperature(value: object) -> float:
    if type(value) not in {int, float}:
        raise DpoConfigError("judge.temperature must be a JSON number")
    temperature = float(value)
    if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
        raise DpoConfigError("judge.temperature must be finite and between 0 and 2")
    return temperature


def _safe_output_name(value: object) -> str:
    if type(value) is str and value.endswith((".", " ")):
        raise DpoConfigError(
            "output_name must be a safe, nonreserved .jsonl basename"
        )
    name = _nonempty_string(value, "output_name")
    lowered = name.casefold()
    windows_name = PureWindowsPath(name)
    unsafe_characters = set('<>:"|?*')

    if (
        not name.endswith(".jsonl")
        or "/" in name
        or "\\" in name
        or ".." in name
        or Path(name).is_absolute()
        or windows_name.is_absolute()
        or bool(windows_name.drive)
        or any(character in unsafe_characters or ord(character) < 32 for character in name)
        or lowered in _RESERVED_OUTPUT_NAMES
        or name.split(".", 1)[0].casefold() in _WINDOWS_DEVICE_NAMES
    ):
        raise DpoConfigError(
            "output_name must be a safe, nonreserved .jsonl basename"
        )
    return name
