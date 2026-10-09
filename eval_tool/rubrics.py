from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .config import EvalConfig
from .prompting import load_prompt_text
from .industrial_rubric import INDUSTRIAL_PROMPT_MARKER


class RubricError(ValueError):
    pass


RUBRIC_VERSIONS = ("v1", "v2", "v3", "v3b", "v4", "v4b")
_PROMPT_ROOT = Path(__file__).resolve().parent.parent / "prompts"
RUBRIC_PROMPTS = {
    version: _PROMPT_ROOT / f"judge_equip_pointwise_{version}.txt"
    for version in RUBRIC_VERSIONS
}
RUBRIC_VERSIONS = RUBRIC_VERSIONS + ("industrial_v1", "doc_v1", "doc_v2", "video_ref_v1")
RUBRIC_PROMPTS["industrial_v1"] = _PROMPT_ROOT / "judge_industrial_pointwise_v1.txt"
# Book / journal pages: the equipment rubrics frame every question as equipment QA and
# fail correct answers about any other content (a formula, a management method).
# doc_v2 keeps v4's graded axes with a domain-neutral gate; doc_v1 is the pass/fail form.
RUBRIC_PROMPTS["doc_v1"] = _PROMPT_ROOT / "judge_doc_pointwise_v1.txt"
RUBRIC_PROMPTS["doc_v2"] = _PROMPT_ROOT / "judge_doc_pointwise_v2.txt"
# Video QA: the judge gets no frames, so it scores against the reference text alone
# (the Video-ChatGPT / lmms-eval practice) instead of describing an image it never saw.
RUBRIC_PROMPTS["video_ref_v1"] = _PROMPT_ROOT / "judge_video_ref_pointwise_v1.txt"


def validate_industrial_rubric(config: EvalConfig, prompt: str | None = None) -> None:
    """Reject domain mismatches before reading images or invoking a judge."""
    industrial = [key for key in config.enabled_datasets
                  if key in config.datasets and config.kind_for(key) == "industrial_describe"]
    selected_prompt = config.judge.pointwise_prompt if prompt is None else prompt
    is_industrial = INDUSTRIAL_PROMPT_MARKER in selected_prompt
    if industrial and not is_industrial:
        raise RubricError("industrial_describe requires rubric industrial_v1; equipment rubrics are incompatible")
    if is_industrial:
        if not industrial or any(config.kind_for(key) != "industrial_describe"
                                 for key in config.enabled_datasets if key in config.datasets):
            raise RubricError("industrial_v1 requires a separate industrial_describe evaluation config")


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
    validate_industrial_rubric(config, prompt)
    judge = replace(config.judge, pointwise_prompt=prompt)
    root = config.out_dir.parent if out_root is None else Path(out_root)
    out_dir = root / f"{config.out_dir.name}_{version}"
    return replace(
        config,
        out_dir=out_dir,
        judge=judge,
        do_pairwise=False if no_pairwise else config.do_pairwise,
    )
