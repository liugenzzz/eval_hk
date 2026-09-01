"""检测框打分器。纯代码，不调任何模型 —— 同一份预测重跑一百遍数字逐位相同。

两个 kind：
    grounding_single   一问一框（指代 -> 坐标）
    grounding_multi    一问多框（框出图中所有的 X）

主指标是【四点偏差】，IoU 系列是副指标。见 docs/模型评估_需求文档.md §3、§4。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

from . import bbox
from .scorers import ScoreContext, register

GROUNDING_SINGLE = "grounding_single"
GROUNDING_MULTI = "grounding_multi"

IOU_GATE = 0.5              # 低于此判定为「定位失败」，不进偏差统计
DEV_THRESHOLDS = (20.0, 50.0)   # 归一化单位；20 ≈ 图宽的 2%


def _empty_metrics() -> Dict[str, Any]:
    return {
        "hit": 0, "iou": 0.0, "located": 0,
        "mae_4pt": pd.NA, "dev_x1": pd.NA, "dev_y1": pd.NA,
        "dev_x2": pd.NA, "dev_y2": pd.NA, "dev_max": pd.NA,
        "iou50": 0, "iou75": 0,
        **{f"dev_le_{int(t)}": pd.NA for t in DEV_THRESHOLDS},
    }


def _score_one(pred_text: Any, gold_text: Any) -> Dict[str, Any]:
    """单框：解析 -> 配上就算偏差，配不上只记定位失败。

    【偏差只在定位成功的样本上算】。模型框到隔壁一辆车，四点偏差可能是 300，
    算进平均值会把整体拉成噪声。所以 IoU >= 0.5 才算偏差，低于的记「定位失败」
    单独计数。报表必须同时出【定位成功率】和【成功样本上的四点偏差】两个数 ——
    只报后者会漂亮得离谱，只报前者又丢了精度信息。
    """
    gold = bbox.parse(gold_text)
    pred = bbox.parse(pred_text)
    row = _empty_metrics()
    for name, cnt in pred.flags.items():
        row[f"fmt_{name}"] = cnt
    row["pred_n"] = len(pred.boxes)
    row["gold_n"] = len(gold.boxes)
    if not gold.ok:
        # 真值都解析不出来，这条样本本身有问题，不该算进任何指标
        row["hit"] = pd.NA
        row["located"] = pd.NA
        row["iou"] = pd.NA
        row["iou50"] = row["iou75"] = pd.NA
        row["gold_bad"] = 1
        return row
    if not pred.ok:
        return row                      # 模型没给出可用的框 = 定位失败

    g = gold.boxes[0]
    # 模型给了多个框而真值只有一个：取和真值最接近的那个来评，多出来的单独计数。
    # 这么做是为了把「多输出」和「找错目标」分开 —— 前者是格式问题，后者是能力问题。
    best = max(pred.boxes, key=lambda b: bbox.iou(b, g))
    if len(pred.boxes) > 1:
        row["extra_boxes"] = len(pred.boxes) - 1

    value = bbox.iou(best, g)
    row["iou"] = value
    row["iou50"] = int(value >= 0.5)
    row["iou75"] = int(value >= 0.75)
    row["located"] = int(value >= IOU_GATE)
    row["hit"] = row["located"]         # 供既有报表把它当二值指标汇总
    if value >= IOU_GATE:
        dev = bbox.deviations(best, g)
        row.update(dev)
        for t in DEV_THRESHOLDS:
            row[f"dev_le_{int(t)}"] = int(dev["dev_max"] <= t)
    return row


def match_boxes(pred: Sequence[bbox.Box], gold: Sequence[bbox.Box],
                iou_gate: float = IOU_GATE) -> List[Tuple[int, int, float]]:
    """匈牙利匹配，IoU 作代价。返回 [(pred_i, gold_j, iou)]，只保留过阈值的对。

    用匈牙利而不是贪心：贪心按 IoU 降序配对，局部最优会让整体少配上一对。
    模型输出没有置信度，无法像 COCO 那样按分数排序，匈牙利在这里是确定且最优的。
    """
    if not pred or not gold:
        return []
    from scipy.optimize import linear_sum_assignment

    cost = [[-bbox.iou(p, g) for g in gold] for p in pred]
    rows, cols = linear_sum_assignment(cost)
    out = []
    for i, j in zip(rows, cols):
        value = -cost[i][j]
        if value >= iou_gate:
            out.append((int(i), int(j), value))
    return out


def _score_multi(pred_text: Any, gold_text: Any) -> Dict[str, Any]:
    """多框：先配对，配上的算偏差，配不上的分漏检和误检。

    数量报两个数：准确率只说「多少张图数对了」，MAE 说「数错时错多少」。
    错 1 个是边界目标的判断，错 10 个是模型压根没在数 —— 完全不同的问题。
    """
    gold = bbox.parse(gold_text)
    pred = bbox.parse(pred_text)
    row: Dict[str, Any] = {}
    for name, cnt in pred.flags.items():
        row[f"fmt_{name}"] = cnt
    n_p, n_g = len(pred.boxes), len(gold.boxes)
    row["pred_n"], row["gold_n"] = n_p, n_g
    if not gold.ok:
        row["gold_bad"] = 1
        for k in ("hit", "precision", "recall", "f1", "count_ok", "count_mae"):
            row[k] = pd.NA
        return row

    row["count_ok"] = int(n_p == n_g)
    row["count_mae"] = abs(n_p - n_g)
    pairs = match_boxes(pred.boxes, gold.boxes)
    tp = len(pairs)
    row["tp"], row["fp"], row["fn"] = tp, n_p - tp, n_g - tp
    row["precision"] = tp / n_p if n_p else 0.0
    row["recall"] = tp / n_g if n_g else 0.0
    denom = row["precision"] + row["recall"]
    row["f1"] = 2 * row["precision"] * row["recall"] / denom if denom else 0.0
    row["hit"] = row["f1"]

    if pairs:
        devs = [bbox.deviations(pred.boxes[i], gold.boxes[j]) for i, j, _ in pairs]
        for key in ("mae_4pt", "dev_x1", "dev_y1", "dev_x2", "dev_y2", "dev_max"):
            row[key] = sum(d[key] for d in devs) / len(devs)
        for t in DEV_THRESHOLDS:
            row[f"dev_le_{int(t)}"] = sum(
                1 for d in devs if d["dev_max"] <= t) / len(devs)
        row["matched_iou"] = sum(v for _, _, v in pairs) / len(pairs)
    else:
        for key in ("mae_4pt", "dev_x1", "dev_y1", "dev_x2", "dev_y2",
                    "dev_max", "matched_iou"):
            row[key] = pd.NA
        for t in DEV_THRESHOLDS:
            row[f"dev_le_{int(t)}"] = pd.NA
    return row


def _apply(data: pd.DataFrame, fn) -> pd.DataFrame:
    scored = data.copy()
    rows = [fn(r.get("prediction", ""), r.get("answer", "")) for _, r in scored.iterrows()]
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    for k in keys:
        scored[k] = [r.get(k, pd.NA) for r in rows]
    return scored


@register(GROUNDING_SINGLE)
def score_grounding_single(data: pd.DataFrame, ctx: ScoreContext) -> pd.DataFrame:
    return _apply(data, _score_one)


@register(GROUNDING_MULTI)
def score_grounding_multi(data: pd.DataFrame, ctx: ScoreContext) -> pd.DataFrame:
    return _apply(data, _score_multi)
