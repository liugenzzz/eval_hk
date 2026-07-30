from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class _SafeFormatMap(dict):
    def __missing__(self, key: str) -> str:
        return ""


@dataclass(frozen=True)
class PromptTemplate:
    template: str

    def render(self, values: Mapping[str, Any]) -> str:
        normalized = _SafeFormatMap({str(k): "" if v is None else str(v) for k, v in values.items()})
        return self.template.format_map(normalized)


def load_prompt_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def load_prompt_files(prompt_files: Mapping[str, str | Path] | None) -> dict[str, str]:
    if not prompt_files:
        return {}
    return {str(key): load_prompt_text(value) for key, value in prompt_files.items() if value}
