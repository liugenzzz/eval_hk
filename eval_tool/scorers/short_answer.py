"""C 组短答案（attribute_qa）：归一化后精确匹配，不中的留给裁判兜底。

**真值优先取 ``metadata.attribute``，不取答案句子。** 构建端的答案是套模板生成的，
同一个属性会写成「深灰色」「是深灰色的。」「深灰色。」几种；拿整句做精确匹配，模型
答对了内容却因为模板不同被判错。更糟的是这个误差**不是对称的**：SFT 模型被训练成
输出这套模板，base 没有 —— 于是 base 会因为**格式**而虚低，测出来的差值里混着格式
差异，和 §10 说的坐标围栏陷阱是同一类问题。

metadata 里的 ``attribute`` 就是这条样本真正要考的内容，没有模板包装。口径同 §7.6
（真值取 metadata，不在评估端另算一份）。

预测那边也剥掉同一套模板前后缀再比。剥不掉的交给裁判判一次同义（阶段 6 的
``judge_synonym`` 兜底还没接，这里先把 ``judge_fallback_needed`` 标出来）。
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from ..compliance import TEXT
from ..synonym import EXACT, HYPERNYM, HYPONYM
from . import CODE, ScoringContext, register
from . import _synonym_fallback

_PUNCT = re.compile(r"[\s，。、；：！？,.;:!?\"'“”‘’()（）\[\]【】]+")
# 构建端答案模板的前后缀：「是…的」「这是…」「该区域内的是…」。只剥这几种固定说法，
# 不做更激进的归一化 —— 剥过头会把「不是深灰色」剥成「深灰色」，把答错判成答对。
_PREFIX = re.compile(r"^(?:图中)?(?:该区域内的)?(?:目标)?(?:是|这是|为)")
_SUFFIX = re.compile(r"的$")


def normalize_answer(text: object) -> str:
    return _PUNCT.sub("", str(text or "")).strip().lower()


def strip_template(text: str) -> str:
    """剥掉答案模板的前后缀。剥完为空就退回原串，别把答案剥没了。"""
    stripped = _SUFFIX.sub("", _PREFIX.sub("", text)).strip()
    return stripped or text


def _gold(row: Any) -> str:
    """真值取 ``metadata.attribute``（无模板包装），但**只在它确实是答案时**。

    ``attribute`` 这个字段在两种任务上含义完全不同：

    - ``attribute_qa``：``attribute="深灰色"`` —— 它**就是答案**，答案句子
      「是深灰色的。」只是套了模板。
    - ``ground_*``：``attribute="穿粉色外套"`` —— 它是**指代用的修饰语**，用来在
      问句里指明是哪一个目标，和答案（一段外观描述）毫无关系。

    无条件优先取它，会把 ground 任务的指代语当成金标，模型答得再对也全判错。判据用
    ``attribute_kind``：构建端只给 ``attribute_qa`` 写这个字段（取值 color / feature），
    ``ground_*`` 一律没有。
    """
    if _present(row, "meta.attribute_kind", "attribute_kind"):
        for column in ("meta.attribute", "attribute"):
            value = row.get(column)
            if value is not None and not _is_na(value) and str(value).strip():
                return normalize_answer(value)
    return strip_template(normalize_answer(row.get("answer", "")))


def _present(row: Any, *columns: str) -> bool:
    for column in columns:
        value = row.get(column)
        if value is not None and not _is_na(value) and str(value).strip():
            return True
    return False


def _is_na(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


@register("short_answer", engine=CODE, answer_form=TEXT)
def score_short_answer(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        gold = _gold(row)
        pred_raw = normalize_answer(row.get("prediction", ""))
        pred = strip_template(pred_raw)
        exact = bool(gold) and gold == pred
        # 宽松匹配是**诊断列，不是指标**：「白色车身的面包车」包含「白色车身」，
        # 按它算会把一堆多说了别的东西的答案算成对的。两个数一起看，差得远说明
        # 模型爱加戏。
        loose = bool(gold) and bool(pred) and (gold in pred or pred in gold)
        rows.append(
            {
                "gold_norm": gold,
                "pred_norm": pred,
                "hit": int(exact),
                "hit_loose": int(loose),
                "empty_prediction": int(not pred_raw),
                "judge_fallback_needed": int(not exact),
            }
        )
    scored = data.copy()
    for column in rows[0] if rows else []:
        scored[column] = [row[column] for row in rows]
    # 「车身是白的」和「白色车身」意思一样、字不一样，代码判不了，交裁判。
    # 短答案没有粒度关系可言，判成同义的三种关系都算命中。
    return _synonym_fallback.apply(
        scored, ctx,
        gold_of=lambda row: str(row.get("gold_norm", "")),
        pred_of=lambda row: str(row.get("prediction", "")),
        promote_relations=(EXACT, HYPERNYM, HYPONYM),
    )
