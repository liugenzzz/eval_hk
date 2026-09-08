"""§10 格式合规、任务串味、截断。微调模型特有的失败模式，公开评测集不会覆盖。

这三个指标降到接近 0 是 SFT 最先体现的效果，涨得最快，也最能早期发现训练配置
有问题。所以它们对**每个数据集**都算，不只是画框那几个。

> **对比 base 时的陷阱**：Qwen3-VL base 会自己给 JSON 加 ```json 围栏。解析器不宽容
> 的话 base 的 Acc 会因**格式**而虚低，报表会好看得离谱 —— 那测的是格式差异不是
> 能力差异。**围栏只计数不扣分**，`format_ok` 不看围栏。

截断这一项目前是**启发式**，见 ``truncation_suspected``：推理侧只写了预测文本，
没有 token 数，精确的「撞上 max_new_tokens」需要改推理的断点续传记录格式。坐标类
答案的截断判得准（截断的 JSON 是没法伪装的），自由文本那一档只是嫌疑。
"""

from __future__ import annotations

import re
from typing import Any, Mapping

import pandas as pd

from .bbox import CODE_FENCE, JSON_REPAIRED, OUT_OF_RANGE, parse_boxes
from .counting import parse_count

# 答案形态。打分器在注册表里声明自己期待哪一种，串味就是「答的不是这一种」。
BOXES = "boxes"
NUMBER = "number"
LISTING = "listing"
TEXT = "text"
YES_NO = "yes_no"
CHOICE = "choice"
ANY = "any"

_COORD_HINT = re.compile(r'"bbox(?:_2d)?"\s*:|\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]')
_TERMINAL = ("。", "！", "？", ".", "!", "?", "」", "”", "}", "]", "）", ")", "；", ";")
# 短于这个长度又没有句末标点的，多半只是答得干脆（「白色车身」），不算截断嫌疑。
TRUNCATION_MIN_CHARS = 30
_CHOICE_HINT = re.compile(r"\b[A-D]\b")


def detect_form(text: object, scale: int = 1000) -> str:
    """看模型实际输出的是哪一种形态。判定顺序：空 → 坐标 → 数 → 文本。"""
    raw = str(text or "").strip()
    if not raw or (isinstance(text, float) and pd.isna(text)):
        return "empty"
    if _COORD_HINT.search(raw) or parse_boxes(raw, scale=scale).ok:
        return BOXES
    if parse_count(raw) is not None and len(raw) <= 12:
        return NUMBER
    return TEXT


def _bleed_kind(expected: str, actual: str) -> str:
    if expected == ANY or actual == "empty" or expected == actual:
        return ""
    if expected == BOXES:
        return "text_for_boxes"      # 问框答文字
    if actual == BOXES:
        return "boxes_for_text"      # 问描述吐坐标
    return ""


def _truncation_evidence(text: str, form: str) -> str:
    """截断的迹象。坐标类判得准，自由文本只是嫌疑。"""
    if not text:
        return ""
    if form == BOXES:
        if text.count("[") != text.count("]") or text.count("{") != text.count("}"):
            return "unbalanced_brackets"
        return ""
    if len(text) >= TRUNCATION_MIN_CHARS and not text.endswith(_TERMINAL):
        return "no_terminal_punctuation"
    return ""


def compliance_row(prediction: object, expected: str, scale: int = 1000) -> dict[str, Any]:
    raw = "" if prediction is None else str(prediction).strip()
    if isinstance(prediction, float) and pd.isna(prediction):
        raw = ""
    actual = detect_form(raw, scale=scale)
    parsed = parse_boxes(raw, scale=scale) if expected == BOXES or actual == BOXES else None

    record: dict[str, Any] = {
        "answer_form": expected,
        "output_form": actual,
        "has_code_fence": int(bool(parsed and CODE_FENCE in parsed.flags)),
        "coord_out_of_range": int(bool(parsed and OUT_OF_RANGE in parsed.flags)),
        "coord_repaired": int(bool(parsed and JSON_REPAIRED in parsed.flags)),
    }

    if expected == BOXES:
        # 围栏不扣分：那是格式差异不是能力差异。越界已裁剪，也只计数。
        format_ok = bool(parsed and parsed.ok)
    elif expected == CHOICE:
        format_ok = bool(_CHOICE_HINT.search(raw))
    elif expected == NUMBER:
        format_ok = parse_count(raw) is not None
    else:
        format_ok = bool(raw)

    bleed = _bleed_kind(expected, actual)
    evidence = _truncation_evidence(raw, actual)
    record.update(
        {
            "format_ok": int(format_ok),
            "task_bleed": int(bool(bleed)),
            "bleed_kind": bleed,
            # 启发式，不是精确的「撞上 max_new_tokens」—— 名字里带 suspected 就是
            # 提醒读表的人别拿它当硬指标。
            "truncation_suspected": int(bool(evidence)),
            "truncation_evidence": evidence,
        }
    )
    return record


def attach_compliance(
    data: pd.DataFrame, expected: str, scale: int = 1000, params: Mapping[str, Any] | None = None
) -> pd.DataFrame:
    if expected == ANY or data.empty or "prediction" not in data.columns:
        return data
    rows = [compliance_row(value, expected, scale=scale) for value in data["prediction"]]
    out = data.copy()
    for column in rows[0]:
        # 打分器自己算过的列优先，合规层不覆盖它。
        if column not in out.columns:
            out[column] = [row[column] for row in rows]
    return out
