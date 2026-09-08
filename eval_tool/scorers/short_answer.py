"""C 组短答案（attribute_qa）：归一化后精确匹配，不中的留给裁判兜底。

``hit`` 只认精确匹配。另外报一列 ``hit_loose``（真值是模型答案的子串，或反过来）
作为诊断量 —— 它不是指标：「白色车身的面包车」包含「白色车身」，宽松口径会把
一堆多说了别的东西的答案算成对的。两个数一起看，差得远说明模型爱加戏。
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ..compliance import TEXT
from . import CODE, ScoringContext, register

_PUNCT = re.compile(r"[\s，。、；：！？,.;:!?\"'“”‘’()（）\[\]【】]+")


def normalize_answer(text: object) -> str:
    return _PUNCT.sub("", str(text or "")).strip().lower()


@register("short_answer", engine=CODE, answer_form=TEXT)
def score_short_answer(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        gold = normalize_answer(row.get("answer", ""))
        pred = normalize_answer(row.get("prediction", ""))
        exact = bool(gold) and gold == pred
        loose = bool(gold) and bool(pred) and (gold in pred or pred in gold)
        rows.append(
            {
                "gold_norm": gold,
                "pred_norm": pred,
                "hit": int(exact),
                "hit_loose": int(loose),
                "empty_prediction": int(not pred),
                # 不精确匹配的丢给裁判判一次同义（阶段 6 接上）。
                "judge_fallback_needed": int(not exact),
            }
        )
    scored = data.copy()
    for column in rows[0] if rows else []:
        scored[column] = [row[column] for row in rows]
    return scored
