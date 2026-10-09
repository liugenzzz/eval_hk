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
    r"\s*(?:是|为|:|：|=|is)?\s*[\*\(\[【]?\s*([A-H])\b",
    r"^\s*\*\*([A-H])\*\*",
)
# 作答以选项字母开头（"B. A man opens the door" / "(C)" / "D：…"）。字母后面必须紧跟
# 标点或结尾：只认大写、不认 "A man" 这种冠词。放在逐字母扫描之前，因为逐字母扫描
# 会把选项文字里的冠词 A 也算进来，一条干净的作答就成了「多个候选」。
_LEADING = re.compile(r"^\s*[\(\[【]?\s*([A-H])\s*(?:[.．、:：)\]】]|$)")


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


# 判断题的对/错词。「没错」「不错」是肯定，里面的「错」不算；英文按字母边界认，
# 否则 "know" / "cannot" 里的 no、"incorrect" 里的 correct 都会被当成作答。
_JUDGE_NEGATIVE = re.compile(r"不正确|不是|不对|错误|错|(?<!是)否|(?<![a-z])(?:false|incorrect|no)(?![a-z])")
_JUDGE_POSITIVE = re.compile(r"正确|对|是|(?<![a-z])(?:true|correct|yes)(?![a-z])")


def _extract_judge_word(text: str) -> str | None:
    normalized = re.sub(r"没错|不错", "对", str(text or "").strip().lower())
    if _JUDGE_NEGATIVE.search(normalized):
        return "B"
    if _JUDGE_POSITIVE.search(normalized):
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

    leading = _LEADING.match(text)
    if leading and leading.group(1) in valid:
        return ChoiceExtraction(choice=leading.group(1), method="leading")

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
