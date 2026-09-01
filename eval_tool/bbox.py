"""检测框的解析与几何指标。纯代码，一行网络请求都没有。

【为什么坐标不交给裁判】框像不像是 IoU 题，不是作文题。同一份预测重跑一百遍，
这里的数字必须逐位相同 —— 有测试守着。

【为什么在归一化空间算偏差】偏差一律在 [0, 1000] 归一化空间计算。像素空间下，
2048 宽的图上差 20px 和 128 宽的图上差 20px 完全不是一回事，混在一起平均没有
意义。归一化空间天然可比，也正好是模型的输出空间。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]

# 坐标系上界。和数据集项目 config 的 coords.scale 对齐。
SCALE = 1000.0

_FENCE = re.compile(r"```(?:json)?\s*|\s*```", re.IGNORECASE)
_NUM = r"-?\d+(?:\.\d+)?"
_BARE_QUAD = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]")


@dataclass
class ParsedBoxes:
    """一次解析的结果。

    boxes 是规范化后的框。flags 记录解析过程中修过什么 —— 这些【只计数不扣分】：
    模型多包一层 ```json 围栏、坐标写反、越界，都不影响它有没有找对目标。
    base 模型尤其爱加围栏，把它算成错误的话，测出来的是格式差异不是能力差异。
    """

    boxes: List[Box] = field(default_factory=list)
    flags: Dict[str, int] = field(default_factory=dict)

    def flag(self, name: str, n: int = 1) -> None:
        self.flags[name] = self.flags.get(name, 0) + n

    @property
    def ok(self) -> bool:
        return bool(self.boxes)


def _norm_one(raw: Sequence[float], out: ParsedBoxes) -> Optional[Box]:
    """规范化一个框：顺序摆正、裁剪到 [0, SCALE]。退化成一条线/一个点就丢掉。"""
    try:
        x1, y1, x2, y2 = (float(v) for v in raw[:4])
    except (TypeError, ValueError):
        out.flag("non_numeric")
        return None
    if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
        out.flag("non_numeric")
        return None
    if x2 < x1 or y2 < y1:
        out.flag("reversed")
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
    if x1 < 0 or y1 < 0 or x2 > SCALE or y2 > SCALE:
        out.flag("out_of_range")
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(SCALE, x2), min(SCALE, y2)
    if x2 <= x1 or y2 <= y1:
        out.flag("degenerate")
        return None
    return (x1, y1, x2, y2)


def _collect(node: Any, found: List[Sequence[float]]) -> None:
    """在任意嵌套结构里找 bbox_2d / bbox / box 字段，或直接是四元组的列表。"""
    if isinstance(node, dict):
        taken = None
        for key in ("bbox_2d", "bbox", "box"):
            v = node.get(key)
            if isinstance(v, (list, tuple)) and len(v) >= 4 and all(
                isinstance(x, (int, float)) for x in v[:4]
            ):
                found.append(v)
                taken = key
                break
        # 【跳过刚取走的那个键】否则下面的递归会把同一个框再收一遍 ——
        # 解析器造出来的重复框会被当成模型的重复输出，误检数直接翻倍。
        for key, v in node.items():
            if key != taken:
                _collect(v, found)
    elif isinstance(node, (list, tuple)):
        if len(node) >= 4 and all(isinstance(x, (int, float)) for x in node[:4]):
            found.append(node)
            return
        for v in node:
            _collect(v, found)


def parse(text: Any) -> ParsedBoxes:
    """从模型输出里解析出框。解析不出返回空 boxes。

    真实输出里这些都会出现，一个都不能崩：空输出、非 JSON、坐标顺序颠倒、
    越界、多包一层 ```json 围栏、返回框数和真值不等、同一目标输出两次。
    """
    out = ParsedBoxes()
    if text is None:
        out.flag("empty")
        return out
    raw = str(text).strip()
    if not raw:
        out.flag("empty")
        return out

    stripped = _FENCE.sub("", raw).strip()
    if stripped != raw:
        out.flag("code_fence")          # 只计数，不扣分

    found: List[Sequence[float]] = []
    for candidate in (stripped, raw):
        try:
            _collect(json.loads(candidate), found)
        except (json.JSONDecodeError, TypeError):
            continue
        if found:
            break
    if not found:
        # JSON 解析不了就退回正则捞四元组 —— 模型常在 JSON 前后加解释性文字
        m = _BARE_QUAD.findall(stripped)
        if m:
            out.flag("regex_fallback")
            found = [tuple(float(v) for v in quad) for quad in m]
    if not found:
        out.flag("unparseable")
        return out

    for quad in found:
        box = _norm_one(quad, out)
        if box is not None:
            out.boxes.append(box)
    if not out.boxes:
        out.flag("unparseable")
    return out


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def deviations(pred: Box, gt: Box) -> Dict[str, float]:
    """四个角点的【有符号】偏差，以及它们的绝对值平均。

    有符号是关键：四个点符号一致说明整体偏移；x1 正而 x2 负说明【系统性框小】，
    反过来是框大。绝对偏差看不出方向，看不出方向就不知道该补什么数据。
    """
    d = {name: float(p) - float(g)
         for name, p, g in zip(("x1", "y1", "x2", "y2"), pred, gt)}
    return {
        **{f"dev_{k}": v for k, v in d.items()},
        "mae_4pt": sum(abs(v) for v in d.values()) / 4.0,
        "dev_max": max(abs(v) for v in d.values()),
    }
