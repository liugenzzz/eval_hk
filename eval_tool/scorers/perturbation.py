"""§11.2 问法扰动：同一个目标换几种说法问，答案框应不应该变。

测的是对问法的过拟合。SFT 数据的问法有限，模型很容易学成「看到某个固定句式才输出
坐标」，换个说法就崩 —— 这个失败模式在真实使用中极常见，而标准评测集完全测不出来
（它们的问法也是固定的）。

**一个组出一行**，不是一个变体出一行：一致性是组的属性，按变体行平均等于把同一个
组数了三遍。
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import pandas as pd

from ..bbox import deviation, iou, parse_boxes
from ..compliance import BOXES
from ..scale import scale_of
from . import CODE, ScoringContext, register

DEFAULT_CONSISTENT_IOU = 0.9
_POINTS = ("x1", "y1", "x2", "y2")


def _group_column(data: pd.DataFrame) -> str | None:
    for column in ("meta.perturb_group", "perturb_group", "meta.derived_from"):
        if column in data.columns:
            return column
    return None


@register("perturbation", engine=CODE, answer_form=BOXES)
def score_perturbation(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    scale, origin = scale_of(data, ctx.params)
    gate = float(ctx.params.get("consistent_iou", DEFAULT_CONSISTENT_IOU))
    group_column = _group_column(data)
    if group_column is None:
        raise ValueError(
            "perturbation 需要分组列（meta.perturb_group）—— 派生集没带，"
            "说明它不是 derive question-perturbation 产出的"
        )

    rows: list[dict[str, Any]] = []
    for group_key, group in data.groupby(group_column, dropna=False, sort=False):
        first = group.iloc[0]
        boxes = []
        parsed_ok = 0
        for prediction in group["prediction"]:
            parsed = parse_boxes(prediction, scale=scale, origin=origin)
            if parsed.ok:
                boxes.append(parsed.boxes[0])
                parsed_ok += 1

        record: dict[str, Any] = {
            "index": str(group_key),
            "model": first.get("model", ""),
            "dataset": first.get("dataset", ""),
            "prediction": first.get("prediction", ""),
            "answer": first.get("answer", ""),
            "question": first.get("question", ""),
            "task_type": first.get("task_type", ""),
            "n_variants": int(len(group)),
            "n_parsed": parsed_ok,
        }
        for column in data.columns:
            if column.startswith("meta.") and column not in record:
                record[column] = first.get(column)

        if len(boxes) < 2:
            # 连两个能解析的框都没有，一致性无从谈起。记 NA 而不是 0：
            # 「换个说法就不输出坐标了」是格式合规率的事，在这里记 0 等于罚两次。
            record.update({
                "pairwise_iou_mean": math.nan, "pairwise_iou_min": math.nan,
                "coord_std_mean": math.nan, "all_pairs_consistent": pd.NA,
                "hit": pd.NA, "fail_reason": "not_enough_parsed_variants",
                **{f"coord_std_{p}": math.nan for p in _POINTS},
            })
            rows.append(record)
            continue

        pairs = [iou(a, b) for a, b in combinations(boxes, 2)]
        stds = {}
        for position, point in enumerate(_POINTS):
            values = [box.as_tuple()[position] for box in boxes]
            mean = sum(values) / len(values)
            stds[point] = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

        consistent = all(value >= gate for value in pairs)
        record.update({
            "pairwise_iou_mean": round(sum(pairs) / len(pairs), 4),
            "pairwise_iou_min": round(min(pairs), 4),
            # 四点各自的标准差，再平均。方差大说明换个说法框就飘。
            "coord_std_mean": round(sum(stds.values()) / 4.0, 4),
            **{f"coord_std_{p}": round(stds[p], 4) for p in _POINTS},
            "all_pairs_consistent": int(consistent),
            "hit": int(consistent),
            "fail_reason": "" if consistent else "inconsistent_across_phrasings",
        })
        # 顺带看一眼这几个变体离真值有多远 —— 一致但一致地错，和一致且对，
        # 是完全不同的结论。
        gold = parse_boxes(first.get("answer", ""), scale=scale, origin=origin)
        if gold.ok:
            devs = [deviation(box, gold.boxes[0], scale=scale).mean4_pct for box in boxes]
            record["dev_mean4_pct_mean"] = round(sum(devs) / len(devs), 4)
            record["gold_iou_mean"] = round(
                sum(iou(box, gold.boxes[0]) for box in boxes) / len(boxes), 4
            )
        else:
            record["dev_mean4_pct_mean"] = math.nan
            record["gold_iou_mean"] = math.nan
        rows.append(record)

    return pd.DataFrame(rows)
