"""A 组单框 / B 组多框的坐标打分器。纯代码，不调任何模型。

验收指标（对外那一个数）
------------------------
::

    达标(单样本) = 解析出框 ∧ IoU ≥ iou_gate ∧ 四点平均偏差 / scale ≤ dev_threshold_pct
    达标率 = 达标样本数 / 全部样本数        验收线 ≥ 75%

三件事写死在这里，改了就不是那个指标了：

1. **分母是全部样本。** 空输出、非 JSON、拒答一律记不达标。只在「定位成功」的
   子集上算达标率，模型可以靠「拿不准就不输出」把分数刷上去。
2. **平均偏差前面串一道 IoU 门。** 单纯的「四点平均 ≤ 5%」有个洞：一个点偏很远、
   另外三点全对时，大偏差会被另外三个点摊薄，一个宽了两倍的框也能判达标。
   IoU 门对正常框几乎零成本，只挡这种病态情形。
3. **偏差只在定位成功的样本上算。** 模型框到隔壁一辆车时四点偏差可能是 300，
   算进平均值会把整体拉成噪声。所以 ``dev_*`` / ``bias_*`` 在 IoU 不达门时留空，
   而「定位成功率」和「成功样本上的偏差」是两个必须分开报的数。

失败的样本再拆三个桶（加起来 = 1 − 达标率），这是这个 KPI 唯一有指导意义的部分：
``malformed`` 格式不合规（训练配置问题，不是能力问题）/ ``localize_fail`` 框到别的
目标去了（补指代消歧、密集场景）/ ``deviation`` 框对了但不够准（补精细边界，
看 ``bias_*`` 定方向）。

两把尺子并列报：``dev_mean4_pct`` 是图幅相对的（验收用这把），``dev_obj`` 是
目标尺寸相对的（小目标的真实精度）。图幅相对的 5% 在 equiv_px<32 的小目标上比
目标本身还大，只看它会把小目标的失败盖过去。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pandas as pd

from ..bbox import Box, ParseResult, deviation, deviation_vs_object, iou, parse_boxes
from ..compliance import BOXES
from ..scale import scale_of
from ..matching import match_boxes
from . import CODE, ScoringContext, register

DEFAULT_IOU_GATE = 0.5
DEFAULT_DEV_THRESHOLD_PCT = 5.0

EXTRA_BOXES = "extra_boxes"      # 单框任务返回了不止一个框，取第一个
GT_UNPARSEABLE = "gt_unparseable"

# 失败桶
BUCKET_OK = ""
BUCKET_MALFORMED = "malformed"
BUCKET_LOCALIZE = "localize_fail"
BUCKET_DEVIATION = "deviation"
BUCKET_GT = "gt_unparseable"

_POINTS = ("x1", "y1", "x2", "y2")


def _thresholds(params: Mapping[str, Any]) -> tuple[float, float]:
    iou_gate = float(params.get("iou_gate", DEFAULT_IOU_GATE))
    dev_pct = float(params.get("dev_threshold_pct", DEFAULT_DEV_THRESHOLD_PCT))
    return iou_gate, dev_pct


def _gt_boxes(row: Mapping[str, Any], scale: int, origin: int) -> ParseResult:
    return parse_boxes(row.get("answer", ""), scale=scale, origin=origin)


def _empty_point_cols(prefix: str) -> dict[str, float]:
    return {f"{prefix}_{point}": math.nan for point in _POINTS}


@register("grounding_single", engine=CODE, answer_form=BOXES)
def score_grounding_single(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    scale, origin = scale_of(data, ctx.params)
    iou_gate, dev_pct = _thresholds(ctx.params)
    threshold = dev_pct / 100.0 * scale

    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        gt = _gt_boxes(row, scale, origin)
        parsed = parse_boxes(row.get("prediction", ""), scale=scale, origin=origin)
        flags = set(parsed.flags)
        if len(parsed.boxes) > 1:
            flags.add(EXTRA_BOXES)

        record: dict[str, Any] = {
            "n_pred_boxes": len(parsed.boxes),
            "parse_ok": bool(parsed.boxes),
            "parse_flags": ",".join(sorted(flags)),
            "gt_ok": bool(gt.boxes),
            **_empty_point_cols("dev"),
            **_empty_point_cols("bias"),
            "dev_mean4": math.nan,
            "dev_max4": math.nan,
            "dev_mean4_pct": math.nan,
            "dev_max4_pct": math.nan,
            "dev_obj": math.nan,
        }

        if not gt.boxes:
            # 真值都解析不出来，这条不是模型的锅，不进任何分母。
            record.update(
                {"iou": math.nan, "localized": pd.NA, "iou50": pd.NA, "iou75": pd.NA,
                 "pass_dev": pd.NA, "pass_dev_strict": pd.NA,
                 "fail_bucket": BUCKET_GT, "hit": pd.NA}
            )
            record["parse_flags"] = ",".join(sorted(flags | {GT_UNPARSEABLE}))
            rows.append(record)
            continue

        gt_box = gt.boxes[0]
        pred_box: Box | None = parsed.boxes[0] if parsed.boxes else None
        overlap = iou(pred_box, gt_box) if pred_box else 0.0
        localized = bool(pred_box) and overlap >= iou_gate

        passed = False
        strict = False
        if localized and pred_box is not None:
            dev = deviation(pred_box, gt_box, scale=scale)
            for point, abs_value, signed in zip(_POINTS, dev.abs4, dev.signed):
                record[f"dev_{point}"] = abs_value
                record[f"bias_{point}"] = signed
            record["dev_mean4"] = dev.mean4
            record["dev_max4"] = dev.max4
            record["dev_mean4_pct"] = dev.mean4_pct
            record["dev_max4_pct"] = dev.max4_pct
            record["dev_obj"] = deviation_vs_object(pred_box, gt_box)
            passed = dev.mean4 <= threshold
            strict = dev.max4 <= threshold

        if passed:
            bucket = BUCKET_OK
        elif not parsed.boxes:
            bucket = BUCKET_MALFORMED
        elif not localized:
            bucket = BUCKET_LOCALIZE
        else:
            bucket = BUCKET_DEVIATION

        record.update(
            {
                "iou": overlap,
                "localized": int(localized),
                "iou50": int(overlap >= 0.5),
                "iou75": int(overlap >= 0.75),
                "pass_dev": int(passed),
                "pass_dev_strict": int(strict),
                "fail_bucket": bucket,
                # hit 就是达标率：报表层每一处汇总、每一条 bootstrap CI 都读这一列，
                # 挂上去 KPI 才不用另起一套汇总。
                "hit": int(passed),
            }
        )
        rows.append(record)

    return _attach(data, rows)


@register("grounding_multi", engine=CODE, answer_form=BOXES)
def score_grounding_multi(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    """多框：先按 IoU 做匈牙利匹配，再在配上的框上算偏差。

    ``hit`` 是**框级**达标率（0~1）：配上且平均偏差达标的框数 / 真值框数。漏检的
    框直接记不达标 —— 否则模型少输出几个框反而能把达标率做高。误检不进分母，
    它由 precision 那一列管。

    不做 COCO 的 ``AP@[.5:.95]``：AP 需要每个框带置信度来排序，模型输出的是纯
    JSON 坐标，没有 score。硬填 1.0 算出来的 AP 是 F1 的一个变形，还多一层解释成本。
    """
    scale, origin = scale_of(data, ctx.params)
    iou_gate, dev_pct = _thresholds(ctx.params)
    threshold = dev_pct / 100.0 * scale

    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        gt = _gt_boxes(row, scale, origin)
        parsed = parse_boxes(row.get("prediction", ""), scale=scale, origin=origin)
        n_gt = len(gt.boxes)
        n_pred = len(parsed.boxes)

        record: dict[str, Any] = {
            "n_gt_boxes": n_gt,
            "n_pred_boxes": n_pred,
            "parse_ok": bool(parsed.boxes),
            "parse_flags": ",".join(sorted(parsed.flags)),
            "gt_ok": bool(gt.boxes),
            **_empty_point_cols("dev"),
            **_empty_point_cols("bias"),
            "dev_mean4": math.nan,
            "dev_mean4_pct": math.nan,
            "dev_obj": math.nan,
            "mean_iou": math.nan,
        }

        if not gt.boxes:
            record.update(
                {"n_matched": 0, "n_missed": 0, "n_spurious": n_pred,
                 "precision": math.nan, "recall": math.nan, "f1": math.nan,
                 "count_correct": pd.NA, "count_error": pd.NA, "count_abs_error": pd.NA,
                 "n_pass_boxes": 0, "pass_dev": math.nan, "hit": pd.NA,
                 "fail_bucket": BUCKET_GT}
            )
            rows.append(record)
            continue

        matching = match_boxes(gt.boxes, parsed.boxes, iou_gate=iou_gate)
        n_matched = len(matching.pairs)
        precision = n_matched / n_pred if n_pred else 0.0
        recall = n_matched / n_gt
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        sums = {point: [] for point in _POINTS}
        signed_sums = {point: [] for point in _POINTS}
        mean4s: list[float] = []
        obj_devs: list[float] = []
        ious: list[float] = []
        n_pass = 0
        for gt_index, pred_index in matching.pairs:
            gt_box, pred_box = gt.boxes[gt_index], parsed.boxes[pred_index]
            dev = deviation(pred_box, gt_box, scale=scale)
            for point, abs_value, signed in zip(_POINTS, dev.abs4, dev.signed):
                sums[point].append(abs_value)
                signed_sums[point].append(signed)
            mean4s.append(dev.mean4)
            obj_devs.append(deviation_vs_object(pred_box, gt_box))
            ious.append(iou(pred_box, gt_box))
            if dev.mean4 <= threshold:
                n_pass += 1

        if mean4s:
            for point in _POINTS:
                record[f"dev_{point}"] = sum(sums[point]) / len(sums[point])
                record[f"bias_{point}"] = sum(signed_sums[point]) / len(signed_sums[point])
            record["dev_mean4"] = sum(mean4s) / len(mean4s)
            record["dev_mean4_pct"] = record["dev_mean4"] / scale * 100.0
            finite_obj = [v for v in obj_devs if not math.isnan(v)]
            record["dev_obj"] = sum(finite_obj) / len(finite_obj) if finite_obj else math.nan
            record["mean_iou"] = sum(ious) / len(ious)

        record.update(
            {
                "n_matched": n_matched,
                "n_missed": len(matching.missed),
                "n_spurious": len(matching.spurious),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                # 数量要两个数：准确率只说「多少张图数对了」，误差说「数错时错多少」。
                # 错 1 个是边界目标的判断，错 10 个是模型压根没在数。
                "count_correct": int(n_pred == n_gt),
                "count_error": n_pred - n_gt,
                "count_abs_error": abs(n_pred - n_gt),
                "n_pass_boxes": n_pass,
                "pass_dev": n_pass / n_gt,
                "hit": n_pass / n_gt,
                "fail_bucket": BUCKET_OK if n_pass == n_gt else (
                    BUCKET_MALFORMED if not parsed.boxes else BUCKET_LOCALIZE
                    if n_matched < n_gt else BUCKET_DEVIATION
                ),
            }
        )
        rows.append(record)

    return _attach(data, rows)


def _attach(data: pd.DataFrame, rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    scored = data.copy()
    if not rows:
        return scored
    for column in rows[0]:
        scored[column] = [row[column] for row in rows]
    return scored
