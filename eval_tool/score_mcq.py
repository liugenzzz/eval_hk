from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .metrics_text import tokenize_for_overlap


@dataclass(frozen=True)
class ChoiceExtraction:
    choice: str | None
    method: str
    ambiguous: bool = False
    no_answer: bool = False


_EXPLICIT_PATTERNS = (
    r"(?:答案|正确答案|最终答案|选项|选择|我选|应选|应该选|answer|final answer|option)"
    r"\s*(?:是|为|:|：|=)?\s*[\*\(\[【]?\s*([A-D])\b",
    r"^\s*\*\*([A-D])\*\*",
)


def _normalize_choice(choice: object) -> str:
    text = str(choice or "").strip().upper()
    m = re.search(r"[A-D]", text)
    return m.group(0) if m else ""


def _normalize_for_text_match(text: object) -> str:
    return "".join(tokenize_for_overlap(text))


def _valid_choices(valid_choices: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(c).upper() for c in valid_choices)


def _extract_explicit(text: str, valid: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    upper = text.upper()
    for pattern in _EXPLICIT_PATTERNS:
        for match in re.finditer(pattern, upper, flags=re.IGNORECASE):
            choice = match.group(1).upper()
            if choice in valid:
                found.append(choice)
    return found


def _extract_independent_letters(text: str, valid: tuple[str, ...]) -> list[str]:
    chars = "".join(re.escape(c) for c in valid)
    pattern = rf"(?<![A-Za-z0-9])([{chars}])(?![A-Za-z0-9])"
    return [m.group(1).upper() for m in re.finditer(pattern, text.upper())]


def _extract_judge_word(text: str) -> str | None:
    normalized = str(text or "").strip().lower()
    negative_patterns = ("不正确", "不是", "不对", "错误", "错", "否", "false", "incorrect", "no")
    positive_patterns = ("正确", "对", "是", "true", "correct", "yes")
    if any(p in normalized for p in negative_patterns):
        return "B"
    if any(p in normalized for p in positive_patterns):
        return "A"
    return None


def _extract_option_text(text: str, options: dict[str, object] | None, valid: tuple[str, ...]) -> str | None:
    if not options:
        return None
    pred_norm = _normalize_for_text_match(text)
    matches: list[str] = []
    for choice in valid:
        value = options.get(choice)
        opt_norm = _normalize_for_text_match(value)
        if opt_norm and opt_norm in pred_norm:
            matches.append(choice)
    unique = list(dict.fromkeys(matches))
    return unique[0] if len(unique) == 1 else None


def _from_found(found: list[str], method: str) -> ChoiceExtraction | None:
    if not found:
        return None
    unique = list(dict.fromkeys(found))
    return ChoiceExtraction(choice=found[0], method=method, ambiguous=len(unique) > 1)


def extract_choice(
    prediction: object,
    valid_choices: Iterable[str] = ("A", "B", "C", "D"),
    options: dict[str, object] | None = None,
    is_judge: bool = False,
) -> ChoiceExtraction:
    text = "" if prediction is None else str(prediction).strip()
    valid = _valid_choices(valid_choices)
    if not text:
        return ChoiceExtraction(choice=None, method="missing", no_answer=True)

    explicit = _from_found(_extract_explicit(text, valid), "explicit")
    if explicit:
        return explicit

    if is_judge:
        judge_choice = _extract_judge_word(text)
        if judge_choice in valid:
            return ChoiceExtraction(choice=judge_choice, method="judge_word")

    independent = _from_found(_extract_independent_letters(text, valid), "letter")
    if independent:
        return independent

    option_choice = _extract_option_text(text, options, valid)
    if option_choice:
        return ChoiceExtraction(choice=option_choice, method="option_text")

    return ChoiceExtraction(choice=None, method="no_answer", no_answer=True)


def score_choice_dataframe(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
    scored = data.copy()
    dataset = dataset.lower()
    valid = ("A", "B") if dataset == "judge" else ("A", "B", "C", "D")
    rows: list[dict[str, object]] = []
    for _, row in scored.iterrows():
        prediction = row.get("prediction", "")
        missing = pd.isna(prediction) or str(prediction).strip() == ""
        options = {choice: row.get(choice, "") for choice in valid if choice in scored.columns}
        extraction = extract_choice(
            prediction,
            valid_choices=valid,
            options=options,
            is_judge=dataset == "judge",
        )
        answer = _normalize_choice(row.get("answer", ""))
        hit = int(extraction.choice == answer) if extraction.choice and answer else 0
        rows.append(
            {
                "extracted_choice": extraction.choice,
                "hit": hit,
                "ambiguous": extraction.ambiguous,
                "no_answer": extraction.no_answer,
                "missing": bool(missing),
                "extraction_method": extraction.method,
                "pred_len": len(tokenize_for_overlap(prediction)),
            }
        )
    for col in rows[0].keys() if rows else []:
        scored[col] = [r[col] for r in rows]
    return scored
