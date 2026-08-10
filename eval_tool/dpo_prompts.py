from __future__ import annotations

from pathlib import Path
from typing import Literal, cast


RubricName = Literal["binary", "v4"]

_PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
_PROMPT_NAMES: dict[tuple[RubricName, bool], str] = {
    ("binary", True): "judge_dpo_binary_visual.txt",
    ("binary", False): "judge_dpo_binary_text.txt",
    ("v4", True): "judge_equip_pointwise_v4.txt",
    ("v4", False): "judge_dpo_v4_text.txt",
}


def dpo_prompt_path(rubric: str, has_images: bool) -> Path:
    """Resolve the fixed DPO prompt for one rubric/modality combination."""
    if rubric not in ("binary", "v4"):
        raise ValueError("DPO rubric must be exactly 'binary' or 'v4'")
    if type(has_images) is not bool:
        raise ValueError("has_images must be a literal boolean")

    name = _PROMPT_NAMES[(cast(RubricName, rubric), has_images)]
    path = (_PROMPT_DIR / name).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DPO judge prompt does not exist: {path}")
    return path


def load_dpo_prompt(rubric: str, has_images: bool) -> str:
    """Load one DPO prompt as nonempty UTF-8 text."""
    path = dpo_prompt_path(rubric, has_images)
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"DPO judge prompt cannot be read as UTF-8: {path}") from exc
    if not prompt:
        raise ValueError(f"DPO judge prompt is empty: {path}")
    return prompt


def _rubric_for_prompt(prompt: str, *, has_images: bool) -> RubricName:
    """Identify a fixed prompt without relying on brittle keyword heuristics."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("DPO judge prompt must be a nonempty string")
    normalized = prompt.strip()
    matches = [
        rubric
        for rubric in ("binary", "v4")
        if normalized == load_dpo_prompt(rubric, has_images)
    ]
    if len(matches) != 1:
        raise ValueError("DPO judge prompt is not the fixed prompt for this modality")
    return matches[0]
