"""选择题 / 判断题：从模型输出里抽出选项字母再比对。默认纯代码，不调模型。"""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from ..score_mcq import extract_choice, score_choice_dataframe
from ..compliance import CHOICE
from . import CODE, ScoringContext, register


@register("choice", engine=CODE, answer_form=CHOICE)
def score_choice(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    # 判断题的合法选项是 A/B，选择题是 A/B/C/D。历史上这件事是按数据集键名
    # ("judge") 判的，所以键名默认继续生效；新数据集应当显式写
    # params.choice_style，别再靠键名取巧。
    style = str(ctx.params.get("choice_style") or ctx.dataset_key)
    return score_choice_dataframe(data, dataset=style)


# 抽取兜底用的提示词：只让裁判认字母，不让它判对错 —— 对错仍由代码比对。
EXTRACT_PROMPT = (
    "你是选择题作答的答案抽取器。下面给出一道选择题的题面和某个模型的作答，"
    "请判断这个作答最终选定了哪一个选项。只输出一个大写选项字母；"
    "如果作答没有明确选定任何选项，或者同时选了多个，只输出 NONE。不要解释。"
)
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _llm_extract(ctx: ScoringContext, question: str, prediction: str, valid: tuple[str, ...]) -> str | None:
    user = f"题面：\n{question}\n\n作答：\n{prediction}\n\n可选字母：{'、'.join(valid)}"
    reply = ctx.judge_client.judge_messages(
        [{"role": "system", "content": EXTRACT_PROMPT}, {"role": "user", "content": user}], max_tokens=64)
    reply = _THINK.sub("", str(reply or "")).strip().upper()
    if "NONE" in reply:
        return None
    letters = re.findall(r"(?<![A-Z])([A-H])(?![A-Z])", reply)
    return letters[-1] if letters and letters[-1] in valid else None


def _meta(row: pd.Series, key: str) -> Any:
    value = row.get(f"meta.{key}")
    return None if value is None or value is pd.NA or isinstance(value, float) and math.isnan(value) else value


@register("choice_circular", engine=CODE, answer_form=CHOICE)
def score_choice_circular(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    """CircularEval：同一题的 N 个轮转版本全部答对，这道题才算对。

    每行是一个轮转版本（``meta.circular_id`` 指回原题，``meta.rotation`` 为 0 的是原题）。
    ``hit`` 只写在原题那一行上，其余行留空 —— 汇总按 ``hit`` 取平均时一道题只算一次，
    置信区间也按题数算。单次作答的准确率另记在 ``vanilla_hit``，两者之差就是位置偏好
    和运气的份量。

    抽字母比普通选择题严：一个作答里出现了两个候选字母（``ambiguous``）按没作答算，
    不取第一个 —— 取第一个等于替模型挑了答案。``params.llm_extract=true`` 且配了裁判时，
    代码抽不出的行交给裁判只认字母；这只会把「没作答」变成某个字母，对错仍由代码比。
    """
    valid = tuple(str(c).upper() for c in ctx.params.get("choices", ("A", "B", "C", "D")))
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        extraction = extract_choice(row.get("prediction"), valid_choices=valid)
        choice = None if extraction.ambiguous else extraction.choice
        rows.append({"extracted_choice": choice, "extraction_method":
                     "ambiguous" if extraction.ambiguous else extraction.method})

    if ctx.params.get("llm_extract") and ctx.judge_client is not None:
        pending = [i for i, r in enumerate(rows) if r["extracted_choice"] is None
                   and str(data.iloc[i].get("prediction") or "").strip()]
        with ThreadPoolExecutor(max_workers=max(1, int(ctx.max_workers))) as pool:
            found = pool.map(lambda i: _llm_extract(ctx, str(data.iloc[i].get("question") or ""),
                                                    str(data.iloc[i].get("prediction") or ""), valid), pending)
            for i, letter in zip(pending, found):
                if letter:
                    rows[i].update(extracted_choice=letter, extraction_method="llm")

    scored = data.copy()
    answers = [re.match(r"\s*([A-H])", str(a or "")) for a in data.get("answer", pd.Series([""] * len(data)))]
    scored["gold_choice"] = [m.group(1) if m else "" for m in answers]
    scored["extracted_choice"] = [r["extracted_choice"] for r in rows]
    scored["extraction_method"] = [r["extraction_method"] for r in rows]
    scored["rotation_hit"] = [int(bool(g) and c == g) for c, g in zip(scored["extracted_choice"], scored["gold_choice"])]

    group = [str(_meta(r, "circular_id") or r.get("index")) for _, r in data.iterrows()]
    rotation = [int(_meta(r, "rotation") or 0) for _, r in data.iterrows()]
    expected = [int(_meta(r, "circular_n") or 0) for _, r in data.iterrows()]
    scored["circular_id"], scored["rotation"] = group, rotation
    by_group: dict[str, list[int]] = {}
    for g, h in zip(group, scored["rotation_hit"]):
        by_group.setdefault(g, []).append(h)
    seen = {g: len(v) for g, v in by_group.items()}
    # 轮转版本缺了（抽样把一题拆开了）就不下结论：少一遍的「全对」不是 CircularEval。
    complete = [not n or seen[g] == n for g, n in zip(group, expected)]
    scored["circular_complete"] = complete
    scored["vanilla_hit"] = [h if r == 0 else pd.NA for h, r in zip(scored["rotation_hit"], rotation)]
    scored["hit"] = [
        (int(all(by_group[g])) if ok else pd.NA) if r == 0 else pd.NA
        for g, r, ok in zip(group, rotation, complete)
    ]
    return scored
