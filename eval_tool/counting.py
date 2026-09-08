"""计数与清单的文本解析。纯代码，不调任何模型。

F 组的答案是一个**数**：``count_class``（「图中有多少辆卡车？」→「3」）和
``inventory_locate`` 轮 1（「图中有哪些清晰可见的目标？」→「有 3 名人员、2 辆卡车
和 1 艘船」）。既有的五组没有一组接得住 —— 按短答案的 Acc 算，数错 1 个和数错 10 个
同罚；按多框算，它压根没输出框。

解析失败的整条记「格式不合规」，**不计入准确率** —— 否则解析器的脆弱会被算成
模型的错。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "两": 2, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

# 「3」「03」「3.0」都认；「第 3 个」这种序数不该被当成计数，靠上下文正则规避。
_ARABIC = re.compile(r"\d+")
_CN_NUMBER = re.compile(r"[零〇一两二三四五六七八九十]+")
_NO_TARGET = re.compile(r"没有|不存在|未(?:发现|找到|看到)|无(?!人机)|一个也没|都没有|0\s*个|none|no\b", re.IGNORECASE)

# 「3名人员」「2 辆卡车」「1艘船」：数字 + 可选量词 + 类别名。
# 量词只是一个可选的单字，不去对量词表 —— 模型把「辆」说成「台」不该算它数错。
_ITEM = re.compile(
    r"(?P<num>\d+|[零〇一两二三四五六七八九十]+)\s*"
    r"(?P<measure>[个只名辆台艘架条把根株棵])?\s*"
    r"(?P<label>[一-鿿A-Za-z][一-鿿A-Za-z0-9]*)"
)
_SPLIT = re.compile(r"[、,，;；]|和|以及|还有")


def cn_to_int(text: str) -> int | None:
    """中文数字转整数。只覆盖 1~99，评估里的计数不会更大。"""
    text = text.strip()
    if not text:
        return None
    if text == "十":
        return 10
    if "十" in text:
        left, _, right = text.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        if left and left not in _CN_DIGITS:
            return None
        if right and right not in _CN_DIGITS:
            return None
        return tens * 10 + ones
    if len(text) == 1:
        return _CN_DIGITS.get(text)
    return None


def to_int(token: str) -> int | None:
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)
    return cn_to_int(token)


def parse_count(text: object) -> int | None:
    """从一句话里抠出计数。抠不出返回 None（记格式不合规，不计入准确率）。

    「图中没有货车」这种否定说法算 0 —— 拒答的另一种形态，口径与 §7.4 一致。
    """
    raw = str(text or "").strip()
    if not raw:
        return None
    match = _ARABIC.search(raw)
    if match:
        return int(match.group(0))
    cn = _CN_NUMBER.search(raw)
    if cn:
        value = cn_to_int(cn.group(0))
        if value is not None:
            return value
    if _NO_TARGET.search(raw):
        return 0
    return None


@dataclass(frozen=True)
class Inventory:
    """清单：类别 -> 数量。``ok=False`` 表示整条解析失败。"""

    items: dict[str, int]
    ok: bool

    @property
    def labels(self) -> set[str]:
        return set(self.items)


def parse_inventory(text: object) -> Inventory:
    """解析「1名人员、2辆卡车和3艘船」这种清单。

    按顿号 / 逗号 / 「和」切分，每段匹配 ``(数字)(量词)?(类别名)``。同一个类别出现
    多次就相加 —— 模型把一类拆成两段说不该算它错。
    """
    raw = str(text or "").strip()
    if not raw:
        return Inventory(items={}, ok=False)
    items: dict[str, int] = {}
    for segment in _SPLIT.split(raw):
        segment = segment.strip()
        if not segment:
            continue
        match = _ITEM.search(segment)
        if not match:
            continue
        count = to_int(match.group("num"))
        label = match.group("label").strip()
        if count is None or not label:
            continue
        items[label] = items.get(label, 0) + count
    if items:
        return Inventory(items=items, ok=True)
    # 一个条目都没抠出来：可能是模型说「图中没有目标」，那是个合法的空清单。
    if _NO_TARGET.search(raw):
        return Inventory(items={}, ok=True)
    return Inventory(items={}, ok=False)


def parse_inventory_gold(value: object) -> Inventory:
    """真值直接取 ``metadata.inventory``，形如 ``["人员x3", "卡车x2"]``。

    不重新数框：构建期已经做过跨任务一致性核对（同一张图上 count_class 说的数、
    detect_class 给的框数、inventory 清单里的数必须相等）。评估端另算一份，就等于
    给同一张图配了两套真值，而那正是构建期那道核对要防的事。
    """
    if value is None:
        return Inventory(items={}, ok=False)
    if isinstance(value, dict):
        return Inventory(items={str(k): int(v) for k, v in value.items()}, ok=True)
    entries: list[str]
    if isinstance(value, (list, tuple)):
        entries = [str(v) for v in value]
    else:
        text = str(value).strip()
        if not text:
            return Inventory(items={}, ok=False)
        if text.startswith("["):
            import json

            try:
                parsed = json.loads(text)
            except ValueError:
                return Inventory(items={}, ok=False)
            return parse_inventory_gold(parsed)
        entries = [part for part in _SPLIT.split(text) if part.strip()]
    items: dict[str, int] = {}
    for entry in entries:
        entry = str(entry).strip()
        match = re.match(r"^(?P<label>.+?)\s*[x×*]\s*(?P<num>\d+)$", entry)
        if not match:
            return Inventory(items={}, ok=False)
        items[match.group("label").strip()] = int(match.group("num"))
    return Inventory(items=items, ok=bool(items))


def count_bin(value: object, small: int = 1, dense: int = 6) -> str:
    """按真值数量分档：单例 / 少量 / 密集。

    TallyQA 的公开结论是准确率随真值数量急剧衰减。合成一个总分，等于让 n=1 那批
    （占比最大也最容易）把 n>=6 的失败盖住。
    """
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    if number <= 0:
        # 真值为 0 的那一路走拒答表，不属于任何数量档 —— 给它分个「单例」会让
        # 单例那一档混进一批根本不是在数数的样本。
        return ""
    if number <= small:
        return "单例"
    if number < dense:
        return "少量"
    return "密集"
