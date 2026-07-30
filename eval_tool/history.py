from __future__ import annotations

import json
from typing import Any

import pandas as pd


def parse_history_cell(value: object) -> list[dict[str, Any]]:
    """The TSV "history" cell holds a JSON list of {"q", "a", "n_img"} for prior turns, or empty."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    turns = []
    for item in parsed:
        if isinstance(item, dict) and "q" in item and "a" in item:
            turns.append({"q": str(item["q"]), "a": str(item["a"]), "n_img": int(item.get("n_img", 0) or 0)})
    return turns


def format_history_for_judge(history: list[dict[str, Any]], question: object) -> str:
    """Render prior turns plus the current question as one text block for judge prompts."""
    if not history:
        return str(question or "")
    lines = []
    for i, turn in enumerate(history, start=1):
        lines.append(f"第{i}轮问：{turn['q']}")
        lines.append(f"第{i}轮答：{turn['a']}")
    lines.append(f"当前问题：{question}")
    return "\n".join(lines)
