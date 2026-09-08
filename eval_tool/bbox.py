"""坐标解析与几何。纯代码，不调任何模型，同一份输入重跑一百遍逐位相同。

坐标一律在 **[0, scale] 归一化空间**（scale 默认 1000，即 Qwen3-VL 的输出空间）里
算，不换算回像素。像素空间下 2048 宽的图上差 20px 和 128 宽的图上差 20px 完全不是
一回事，混在一起平均没有意义；归一化空间天然可比。

换算公式对齐数据构建端的 ``core/coords.py``：那边 ``pixel_to_bbox2d`` 用
``round(clamp(v, 0, img) / img * scale)`` 落到 0~scale 的整数，并把结果 clamp 到
``[origin, scale + origin]``。所以 **scale 本身是合法取值**（一个贴着右边缘的框
x2 就是 1000），越界判据是 ``> scale``，不是 ``>= scale``。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# 解析标记。这些只计数，不扣分 —— 对比 base 时尤其重要：Qwen3-VL base 会自己给
# JSON 加 ```json 围栏，解析器不宽容的话 base 的分会因【格式】而虚低，报表会好看
# 得离谱，但那测的是格式差异不是能力差异。
EMPTY = "empty"                  # 模型什么都没输出
UNPARSEABLE = "unparseable"      # 连一个框都没抠出来
JSON_REPAIRED = "json_repaired"  # 不是合法 JSON，靠正则救回来的
CODE_FENCE = "code_fence"        # 多包了一层 ```json 围栏
REORDERED = "reordered"          # x2 < x1 或 y2 < y1，已规范化
OUT_OF_RANGE = "out_of_range"    # 坐标越界，已裁剪
BAD_BOX = "bad_box"              # 某个元素凑不出四个数，已丢弃
DUPLICATE = "duplicate"          # 同一个框输出了两次

_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$", re.DOTALL)
_BBOX_FIELD = re.compile(r'"bbox(?:_2d)?"\s*:\s*\[([^\]]*)\]')
_BARE_ARRAY = re.compile(r"\[\s*(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){3})\s*\]")
_LABEL_FIELD = re.compile(r'"label"\s*:\s*"([^"]*)"')


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class ParseResult:
    boxes: tuple[Box, ...]
    flags: frozenset[str]

    @property
    def ok(self) -> bool:
        return bool(self.boxes)

    @property
    def flag_text(self) -> str:
        return ",".join(sorted(self.flags))


def parse_boxes(text: object, scale: int = 1000, origin: int = 0) -> ParseResult:
    """从模型输出里抠出全部框。真实输出里的退化情形一个都不许崩。"""
    flags: set[str] = set()
    raw = "" if text is None else str(text)
    if isinstance(text, float) and math.isnan(text):
        raw = ""
    if not raw.strip():
        return ParseResult(boxes=(), flags=frozenset({EMPTY, UNPARSEABLE}))

    body = raw.strip()
    fence = _FENCE.match(body)
    if fence:
        flags.add(CODE_FENCE)
        body = fence.group(1).strip()

    raw_boxes = _load_json_boxes(body)
    if raw_boxes is None:
        raw_boxes = _regex_boxes(body)
        if raw_boxes:
            flags.add(JSON_REPAIRED)

    boxes: list[Box] = []
    seen: set[tuple[float, float, float, float]] = set()
    for coords, label in raw_boxes or []:
        box = _normalize_box(coords, label, scale, origin, flags)
        if box is None:
            continue
        if box.as_tuple() in seen:
            flags.add(DUPLICATE)
        seen.add(box.as_tuple())
        # 重复框保留不去重：模型把同一个目标输出两次是一种真实的失败方式，
        # 悄悄去重就等于替它擦掉，多框任务里那一个应当被记成误检。
        boxes.append(box)

    if not boxes:
        flags.add(UNPARSEABLE)
    return ParseResult(boxes=tuple(boxes), flags=frozenset(flags))


def _load_json_boxes(body: str) -> list[tuple[Sequence[Any], str]] | None:
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return None
    return _collect(data)


def _collect(data: Any) -> list[tuple[Sequence[Any], str]]:
    out: list[tuple[Sequence[Any], str]] = []
    if isinstance(data, dict):
        coords = data.get("bbox_2d", data.get("bbox"))
        if isinstance(coords, (list, tuple)):
            out.append((coords, str(data.get("label", "") or "")))
        else:
            for value in data.values():
                out.extend(_collect(value))
    elif isinstance(data, (list, tuple)):
        # 裸数组 [x1, y1, x2, y2]
        if len(data) == 4 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in data):
            out.append((list(data), ""))
        else:
            for item in data:
                out.extend(_collect(item))
    return out


def _regex_boxes(body: str) -> list[tuple[Sequence[Any], str]]:
    """JSON 解析失败时的兜底：截断、少个引号、括号没配对都会走到这里。"""
    out: list[tuple[Sequence[Any], str]] = []
    labels = _LABEL_FIELD.findall(body)
    for i, match in enumerate(_BBOX_FIELD.finditer(body)):
        parts = [p.strip() for p in match.group(1).split(",") if p.strip()]
        out.append((parts, labels[i] if i < len(labels) else ""))
    if out:
        return out
    for i, match in enumerate(_BARE_ARRAY.finditer(body)):
        parts = [p.strip() for p in match.group(1).split(",")]
        out.append((parts, labels[i] if i < len(labels) else ""))
    return out


def _normalize_box(
    coords: Sequence[Any], label: str, scale: int, origin: int, flags: set[str]
) -> Box | None:
    if len(coords) != 4:
        flags.add(BAD_BOX)
        return None
    values: list[float] = []
    for item in coords:
        try:
            value = float(item)
        except (TypeError, ValueError):
            flags.add(BAD_BOX)
            return None
        if not math.isfinite(value):
            flags.add(BAD_BOX)
            return None
        values.append(value)

    x1, y1, x2, y2 = values
    if x2 < x1 or y2 < y1:
        flags.add(REORDERED)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
    lo, hi = float(origin), float(scale + origin)
    if any(v < lo or v > hi for v in (x1, y1, x2, y2)):
        flags.add(OUT_OF_RANGE)
        x1, y1, x2, y2 = (min(max(v, lo), hi) for v in (x1, y1, x2, y2))
    return Box(x1=x1, y1=y1, x2=x2, y2=y2, label=str(label or ""))


def iou(a: Box, b: Box) -> float:
    inter_w = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    inter_h = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = inter_w * inter_h
    union = a.area + b.area - inter
    if union <= 0:
        # 两个都是零面积的退化框：位置一样才算重合，否则算不重合。
        return 1.0 if a.as_tuple() == b.as_tuple() else 0.0
    return inter / union


@dataclass(frozen=True)
class Deviation:
    """四点偏差。signed 是 pred - gt，abs 是它的绝对值。

    有符号的那一组最有诊断价值：四个点符号一致 = 整体偏移；
    ``bias_x1 > 0`` 且 ``bias_x2 < 0`` = 系统性框小，反过来 = 系统性框大。
    绝对偏差看不出方向，看不出方向就不知道该补什么数据。
    """

    signed: tuple[float, float, float, float]
    scale: int

    @property
    def abs4(self) -> tuple[float, float, float, float]:
        return tuple(abs(v) for v in self.signed)  # type: ignore[return-value]

    @property
    def mean4(self) -> float:
        return sum(self.abs4) / 4.0

    @property
    def max4(self) -> float:
        return max(self.abs4)

    @property
    def mean4_pct(self) -> float:
        """占图幅的百分比 —— 验收指标（偏差 ≤ 5%）读的就是这个数。"""
        return self.mean4 / self.scale * 100.0

    @property
    def max4_pct(self) -> float:
        return self.max4 / self.scale * 100.0


def deviation(pred: Box, gt: Box, scale: int = 1000) -> Deviation:
    return Deviation(
        signed=(pred.x1 - gt.x1, pred.y1 - gt.y1, pred.x2 - gt.x2, pred.y2 - gt.y2),
        scale=scale,
    )


def deviation_vs_object(pred: Box, gt: Box) -> float:
    """相对【目标自身尺寸】的平均四点偏差（比例，1.0 = 偏了一个框宽/高）。

    图幅相对的 5% 在小目标上是失效的：equiv_px<32 的目标边长只占图幅 3%，
    四个点各偏 5% 时框已经完全没框住目标，但按图幅尺算它「达标」。这把尺子
    并列报出来，小目标那一档才看得见真实精度。gt 退化成零宽/零高时返回 nan。
    """
    if gt.width <= 0 or gt.height <= 0:
        return math.nan
    dx = (abs(pred.x1 - gt.x1) + abs(pred.x2 - gt.x2)) / 2.0 / gt.width
    dy = (abs(pred.y1 - gt.y1) + abs(pred.y2 - gt.y2)) / 2.0 / gt.height
    return (dx + dy) / 2.0


def mean_or_nan(values: Iterable[float]) -> float:
    items = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(items) / len(items) if items else math.nan
