"""空间方位二选一：「A 在 B 的左侧还是右侧」「A 在 B 的上方还是下方」。

真值在 ``meta.relation``（左侧 / 右侧 / 上方 / 下方），``meta.relation_axis`` 指明问的
是哪条轴（lr / ud）。只认那条轴上的方位词：两个方向都出现、又没有否定掉其中一个时算
没答出来（``parse_ok`` 为假），不猜。

读不准的一种写法要知道：模型把主语和参照物对调着答（问「人员在手表的哪侧」，答
「手表在人员的左侧」），这里按字面读成「左侧」。这种答法少见，但它会被判错。
"""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

from ..compliance import TEXT
from . import CODE, ScoringContext, register

# 方位词写全形式；单字「上」「下」到处都是（上衣、身上、以下），不单独认。
_DIRECTIONS = {
    "lr": {
        "左侧": r"左(?:侧|边|面|方|手边?)|偏左|靠左|左(?=\s*[。．.，,！!；;]|\s*$)|(?<![a-z])left(?![a-z])",
        "右侧": r"右(?:侧|边|面|方|手边?)|偏右|靠右|右(?=\s*[。．.，,！!；;]|\s*$)|(?<![a-z])right(?![a-z])",
    },
    "ud": {
        "上方": r"上(?:方|面|边|侧|部|头|空)|之上|偏上|靠上|(?<![a-z])(?:above|over)(?![a-z])",
        "下方": r"下(?:方|面|边|侧|部|头)|之下|底下|偏下|靠下|(?<![a-z])(?:below|under(?:neath)?)(?![a-z])",
    },
}
_ALL = "|".join(p for axis in _DIRECTIONS.values() for p in axis.values())
# 「左右」是约数（「十米左右」），「上下」同理；复述问题（「左侧还是右侧」）不是答案。
_NOISE = re.compile(rf"左右|上下|(?:{_ALL})\s*还是\s*(?:{_ALL})", re.I)
# 被否定掉的方位：「不在柜子的左侧，而在右侧」。
_NEGATED = re.compile(rf"(?:不在|不是|并非|并不在|而非)[^，。；,;！!？?\n]{{0,8}}?(?:{_ALL})", re.I)


# 长回答先说结论再逐条分析，分析里两个方向都会提到（「足球位于画面左侧……鞋在足球
# 右侧」）。按结论句、首句、全文的顺序读，第一个只含一个方向的就是答案。
_CONCLUSION = re.compile(r"(?:因此|所以|综上所述?|总之|结论)[\s*:：，,]*([^\n。]+)")


def _spans(text: str) -> list[str]:
    spans = [m.group(1) for m in _CONCLUSION.finditer(text)][::-1]
    first = re.split(r"[。\n]", text.strip(), maxsplit=1)[0]
    return [*spans, first, text]


def parse_relation(value: Any, axis: str) -> str | None:
    if value is None or value is pd.NA or isinstance(value, float) and math.isnan(value):
        return None
    patterns = _DIRECTIONS.get(axis)
    if not patterns:
        return None
    for span in _spans(str(value).replace("**", "")):
        text = _NEGATED.sub(" ", _NOISE.sub(" ", span))
        found = [name for name, pattern in patterns.items() if re.search(pattern, text, re.I)]
        if len(found) == 1:
            return found[0]
    return None


def _meta(row: pd.Series, key: str) -> Any:
    value = row.get(f"meta.{key}")
    return None if value is None or value is pd.NA or isinstance(value, float) and math.isnan(value) else value


@register("spatial_relation", engine=CODE, answer_form=TEXT)
def score_spatial_relation(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    rows = []
    for _, row in data.iterrows():
        axis = str(_meta(row, "relation_axis") or "").strip()
        gold = str(_meta(row, "relation") or "").strip()
        if axis not in _DIRECTIONS or gold not in _DIRECTIONS[axis]:
            raise ValueError(f"spatial_relation 缺少真值：relation={gold!r} relation_axis={axis!r}")
        pred = parse_relation(row.get("prediction"), axis)
        rows.append({"gold_relation": gold, "pred_relation": pred or "",
                     "parse_ok": pred is not None, "hit": int(pred == gold)})
    scored = data.copy()
    for key in ("gold_relation", "pred_relation", "parse_ok", "hit"):
        scored[key] = [r[key] for r in rows]
    return scored
