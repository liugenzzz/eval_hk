from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import EvalConfig
from .prompting import load_prompt_text


class RubricError(ValueError):
    pass


RUBRIC_VERSIONS = ("v1", "v2", "v3", "v3b", "v4", "v4b")
_PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts"
RUBRIC_PROMPTS = {
    version: _PROMPT_ROOT / f"judge_equip_pointwise_{version}.txt"
    for version in RUBRIC_VERSIONS
}


def rubric_prompt_path(version: str) -> Path:
    try:
        path = RUBRIC_PROMPTS[str(version)]
    except KeyError as exc:
        raise RubricError(f"unknown rubric: {version}") from exc
    if not path.is_file():
        raise RubricError(f"rubric prompt not found: {path}")
    return path


def validate_rubric_list(versions: list[str] | tuple[str, ...]) -> list[str]:
    normalized = [str(version).strip() for version in versions]
    if not normalized or any(not version for version in normalized):
        raise RubricError("at least one rubric is required")
    if len(set(normalized)) != len(normalized):
        raise RubricError("duplicate rubrics are not allowed")
    for version in normalized:
        rubric_prompt_path(version)
    return normalized


def apply_rubric(
    config: EvalConfig,
    version: str,
    out_root: str | Path | None = None,
    no_pairwise: bool = False,
) -> EvalConfig:
    prompt = load_prompt_text(rubric_prompt_path(version))
    judge = replace(config.judge, pointwise_prompt=prompt)
    root = config.out_dir.parent if out_root is None else Path(out_root)
    out_dir = root / f"{config.out_dir.name}_{version}"
    return replace(
        config,
        out_dir=out_dir,
        judge=judge,
        do_pairwise=False if no_pairwise else config.do_pairwise,
    )
