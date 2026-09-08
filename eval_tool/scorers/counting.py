"""F 组计数 / 清单，以及拒答表。纯代码，不调任何模型。

§7 的三条口径写死在这里：

1. **三个数一起报，缺一个都不够**：精确命中率（主指标）、计数 MAE、计数偏向。
   准确率只说「多少条数对了」，MAE 说「数错时错多少」—— 错 1 个是边界目标的判断，
   错 10 个是模型压根没在数。**偏向是第三个必须有的数**：SFT 之后常见的退化是
   「见到密集场景就报一个大概的整数」，那时准确率和 MAE 会一起变差，但只有符号
   说得清它是往多了猜还是往少了猜，而这两种要补的数据完全不同。
2. **真值取 metadata，不重新数框**（``meta.count`` / ``meta.inventory``）。构建期已经
   做过跨任务一致性核对，评估端另算一份就等于给同一张图配了两套真值。
3. **``counting == "zero"`` 的样本不计入 F 组准确率**，并进拒答表：它和 exist_negative
   考的是同一件事（模型会不会「被问就一定有」），只是输出形态从「没有」换成「0」。
   两种形态在拒答表里分行列 —— 模型可能答得出「没有」，却在被要求给一个数时编出「1」。

分档（单例 / 少量 / 密集）和问法（exact / visible）都只是打在样本上的列，怎么拆是
报表层的事。``visible`` 那一路的真值带主观限定，准确率天然低于 ``exact``，
**checkpoint 之间的对比只用 exact 那一路**。
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ..counting import Inventory, count_bin, parse_count, parse_inventory, parse_inventory_gold
from . import CODE, ScoringContext, register

ZERO = "zero"


def _meta(row: Any, name: str) -> Any:
    for column in (f"meta.{name}", name):
        if column in row and row[column] is not None and not _is_na(row[column]):
            return row[column]
    return None


def _is_na(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _gold_count(row: Any) -> int | None:
    value = _meta(row, "count")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    # metadata 没带 count 时退回从标准答案里解析 —— 只在旧数据上会走到。
    return parse_count(row.get("answer", ""))


@register("counting", engine=CODE)
def score_counting(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    dense = int(ctx.params.get("dense_from", 6))
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        gold = _gold_count(row)
        predicted = parse_count(row.get("prediction", ""))
        counting = str(_meta(row, "counting") or "")
        is_zero_route = counting == ZERO or gold == 0

        record: dict[str, Any] = {
            "gold_count": gold if gold is not None else pd.NA,
            "pred_count": predicted if predicted is not None else pd.NA,
            "counting": counting,
            "count_bin": count_bin(gold, dense=dense) if gold is not None else "",
            "parse_ok": predicted is not None,
            # zero 一路并进拒答表，不计入 F 组准确率。
            "refusal_route": int(is_zero_route),
            "refusal_form": "count_zero" if is_zero_route else "",
        }

        if gold is None or predicted is None:
            record.update(
                {"count_error": pd.NA, "count_abs_error": pd.NA, "hit": pd.NA,
                 "refusal_hit": pd.NA, "said_yes": pd.NA,
                 "fail_reason": "unparseable_prediction" if gold is not None else "missing_gold"}
            )
            # 解析不出来的整条记格式不合规。计入准确率就等于把解析器的脆弱算成模型的错，
            # 但它也不能凭空消失 —— parse_ok 那一列就是给这件事用的。
            rows.append(record)
            continue

        error = predicted - gold
        record.update(
            {
                "count_error": error,          # 有符号：正 = 系统性多报，负 = 系统性漏报
                "count_abs_error": abs(error),
                "fail_reason": "" if error == 0 else "wrong_count",
            }
        )
        if is_zero_route:
            record.update(
                {
                    "hit": pd.NA,                       # 不进 F 组准确率
                    "refusal_hit": int(predicted == 0),  # 进拒答表
                    "said_yes": int(predicted != 0),     # yes 偏置率的分子
                }
            )
        else:
            record.update({"hit": int(error == 0), "refusal_hit": pd.NA, "said_yes": pd.NA})
        rows.append(record)

    return _attach(data, rows)


@register("inventory", engine=CODE)
def score_inventory(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    """§7.5 清单：两层判定，**不许合成一个分**。

    1. **类别集合** P / R / F1 —— 漏报一个类别 vs 编出一个图里没有的类别
    2. **数量** —— 只在类别对上的那些项上算精确命中率与 MAE

    类别报全了但数全错，和类别漏一半但数都对，是完全不同的问题，合成之后两者可能
    得同一个分。
    """
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        gold = parse_inventory_gold(_meta(row, "inventory"))
        if not gold.ok:
            gold = parse_inventory(row.get("answer", ""))
        parsed = parse_inventory(row.get("prediction", ""))
        rows.append(_inventory_record(gold, parsed))
    return _attach(data, rows)


def _inventory_record(gold: Inventory, parsed: Inventory) -> dict[str, Any]:
    record: dict[str, Any] = {
        "parse_ok": parsed.ok,
        "gold_labels": ",".join(sorted(gold.labels)),
        "pred_labels": ",".join(sorted(parsed.labels)),
    }
    if not gold.ok:
        record.update(_blank_inventory("missing_gold"))
        return record
    if not parsed.ok:
        # 解析不出来的整条记格式不合规，不计入准确率 —— 否则解析器的脆弱会被算成
        # 模型的错。集合级的漏报仍然如实记：一个类别都没报出来就是全漏。
        record.update(_blank_inventory("unparseable_prediction"))
        record.update({"set_recall": 0.0, "set_precision": math.nan, "set_f1": 0.0,
                       "n_gold_labels": len(gold.labels), "n_pred_labels": 0,
                       "n_labels_matched": 0, "hit": 0})
        return record

    matched = gold.labels & parsed.labels
    precision = len(matched) / len(parsed.labels) if parsed.labels else 0.0
    recall = len(matched) / len(gold.labels) if gold.labels else math.nan
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    exact_counts = [int(gold.items[label] == parsed.items[label]) for label in sorted(matched)]
    errors = [parsed.items[label] - gold.items[label] for label in sorted(matched)]
    record.update(
        {
            "n_gold_labels": len(gold.labels),
            "n_pred_labels": len(parsed.labels),
            "n_labels_matched": len(matched),
            "set_precision": precision,
            "set_recall": recall,
            "set_f1": f1,
            "labels_missed": ",".join(sorted(gold.labels - parsed.labels)),
            "labels_spurious": ",".join(sorted(parsed.labels - gold.labels)),
            # 数量只在类别对上的项上算，和集合指标分开报。
            "count_hit_rate": sum(exact_counts) / len(exact_counts) if exact_counts else math.nan,
            "count_abs_error": sum(abs(e) for e in errors) / len(errors) if errors else math.nan,
            "count_error": sum(errors) / len(errors) if errors else math.nan,
            # hit 挂集合 F1：报表层的默认汇总读这一列。数量那两个数单独看，
            # 合成一个分正是这一节明令禁止的事。
            "hit": f1,
            "fail_reason": "" if f1 == 1.0 and all(exact_counts) else "wrong_inventory",
        }
    )
    return record


def _blank_inventory(reason: str) -> dict[str, Any]:
    return {
        "n_gold_labels": pd.NA, "n_pred_labels": pd.NA, "n_labels_matched": pd.NA,
        "set_precision": math.nan, "set_recall": math.nan, "set_f1": math.nan,
        "labels_missed": "", "labels_spurious": "",
        "count_hit_rate": math.nan, "count_abs_error": math.nan, "count_error": math.nan,
        "hit": pd.NA, "fail_reason": reason,
    }


@register("exist_negative", engine=CODE)
def score_exist_negative(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    """拒答表：拒答准确率 + yes 偏置率。

    负样本是**易混类别**的 hard negative（图里有卡车，问有没有货车），相当于 POPE 的
    Adversarial 档，难度远高于 Random 档 —— **不要拿 POPE 的公开数字横向比**。

    yes 偏置率（模型有多倾向于说「有」）单独报：SFT 之后 yes 偏置变高是常见退化，
    而它在总准确率里是看不见的。
    """
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        polarity = str(_meta(row, "polarity") or "").strip().lower()
        gold_yes = _answer_says_yes(row.get("answer", "")) if polarity not in {"positive", "negative"} else polarity == "positive"
        pred_yes = _answer_says_yes(row.get("prediction", ""))
        rows.append(
            {
                "polarity": polarity or ("positive" if gold_yes else "negative"),
                "gold_yes": pd.NA if gold_yes is None else int(gold_yes),
                "said_yes": pd.NA if pred_yes is None else int(pred_yes),
                "refusal_form": "exist_no",
                "hit": pd.NA if (gold_yes is None or pred_yes is None) else int(gold_yes == pred_yes),
                "refusal_hit": pd.NA if (gold_yes or gold_yes is None or pred_yes is None) else int(pred_yes is False),
                "parse_ok": pred_yes is not None,
            }
        )
    return _attach(data, rows)


_YES = ("有", "是", "存在", "能找到", "看到", "yes")
_NO = ("没有", "不存在", "未发现", "找不到", "看不到", "无", "不是", "no")


def _answer_says_yes(text: object) -> bool | None:
    raw = str(text or "").strip().lower()
    if not raw:
        return None
    # 先判否定：「没有」里含「有」，顺序反了会把每一条都判成 yes。
    for token in _NO:
        if token in raw:
            return False
    for token in _YES:
        if token in raw:
            return True
    number = parse_count(raw)
    if number is not None:
        return number > 0
    return None


def _attach(data: pd.DataFrame, rows: list[dict[str, Any]]) -> pd.DataFrame:
    scored = data.copy()
    if not rows:
        return scored
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    for key in keys:
        scored[key] = [row.get(key, pd.NA) for row in rows]
    return scored
