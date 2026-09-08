"""类别表与上下位判定。纯代码，不调任何模型。

算法移植自数据构建端的 ``core/classes.py::_detect_confusable``。那边的 hypernym
**不是一张静态表**，是从类别名单按两条判据现算的：

1. **包含关系**：``三轮车`` ⊂ ``遮阳三轮车``、``人员`` ⊂ ``军事人员`` —— 多半是上下位词。
2. **等长且只差一个字**：``切管器`` vs ``切管机``、``压接钳`` vs ``压管钳`` —— 这类
   并列的不同东西靠视觉几乎不可能可靠区分，标成「易混」但不是上下位。

构建端的 hypernym 组是**对称**的（互相加入对方），它只说「这两个名字有包含关系」，
不区分谁是上位。评估要判「模型答的是不是金标的上位词」，需要方向 —— 方向从名字
长度就能定：短的那个是上位词。于是判定从三档变四档：

===========  ====================  ==========================================
判定          例（金标 → 模型答）    含义
===========  ====================  ==========================================
exact         遮阳三轮车 → 遮阳三轮车  精确命中，**主指标**
hypernym      遮阳三轮车 → 三轮车     答粗了。安全但是退化，单独计数，不许并进主指标
hyponym       三轮车 → 遮阳三轮车     答细了，很可能在幻觉一个看不见的属性，单独计数
other         三轮车 → 卡车          错误
===========  ====================  ==========================================

把 hyponym 从 other 里拆出来是有意的：模型把「三轮车」答成「遮阳三轮车」是在编一个
它看不清的属性，和答成「卡车」是两种病，混在一起就不知道该补哪种数据。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

EXACT = "exact"
HYPERNYM = "hypernym"
HYPONYM = "hyponym"
OTHER = "other"
OFF_TABLE = "off_table"      # 模型自创了一个类别表里没有的词


def normalize(name: object) -> str:
    return str(name or "").strip().lower().replace(" ", "").replace("　", "")


def one_char_apart(a: str, b: str) -> bool:
    """等长且恰好只有一个字符不同。口径与构建端一致。"""
    if len(a) != len(b) or a == b:
        return False
    return sum(1 for x, y in zip(a, b) if x != y) == 1


@dataclass(frozen=True)
class ClassTable:
    """id -> 类别名。名称查找一律走 normalize 后的键。"""

    id2name: Mapping[int, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_names", tuple(str(n).strip() for n in self.id2name.values()))
        object.__setattr__(self, "_norm2name", {normalize(n): str(n).strip() for n in self.id2name.values()})
        # 长的排前面：「遮阳三轮车」要先于「三轮车」被匹配到，否则模型答对了也会
        # 被抠成上位词。
        object.__setattr__(
            self, "_by_length", tuple(sorted((normalize(n) for n in self.id2name.values()), key=len, reverse=True))
        )

    @property
    def names(self) -> tuple[str, ...]:
        return self._names  # type: ignore[attr-defined]

    @property
    def count(self) -> int:
        return len(self.id2name)

    def contains(self, name: object) -> bool:
        return normalize(name) in self._norm2name  # type: ignore[attr-defined]

    def canonical(self, name: object) -> str | None:
        return self._norm2name.get(normalize(name))  # type: ignore[attr-defined]

    def find_in_text(self, text: object) -> str | None:
        """从一句话里抠出类别名（「该区域内的是面包车。」→ 面包车）。

        最长匹配优先，所以「遮阳三轮车」不会被抠成「三轮车」。一个都找不到时返回
        None —— 那是模型自创了词，§6 说这时才丢给裁判判一次是不是同义。
        """
        haystack = normalize(text)
        if not haystack:
            return None
        for candidate in self._by_length:  # type: ignore[attr-defined]
            if candidate and candidate in haystack:
                return self._norm2name[candidate]  # type: ignore[attr-defined]
        return None

    def find_all_in_text(self, text: object) -> tuple[str, ...]:
        """一句话里提到的**全部**类别名，最长匹配且不重叠。

        「一辆遮阳三轮车停在卡车旁边」→ (遮阳三轮车, 卡车)，不会因为「遮阳三轮车」
        里含「三轮车」而多数出一个类别 —— CHAIR 幻觉率按类别计数，多数一个就是
        凭空多一次幻觉。
        """
        haystack = normalize(text)
        if not haystack:
            return ()
        found: list[tuple[int, str]] = []
        occupied = [False] * len(haystack)
        for candidate in self._by_length:  # type: ignore[attr-defined]
            if not candidate:
                continue
            start = haystack.find(candidate)
            while start >= 0:
                end = start + len(candidate)
                if not any(occupied[start:end]):
                    for i in range(start, end):
                        occupied[i] = True
                    found.append((start, self._norm2name[candidate]))  # type: ignore[attr-defined]
                start = haystack.find(candidate, start + 1)
        seen: dict[str, None] = {}
        for _, name in sorted(found):
            seen.setdefault(name, None)
        return tuple(seen)

    def relation(self, gold: object, pred: object) -> str:
        """四档判定。pred 不在类别表里时返回 OFF_TABLE。"""
        gold_norm = normalize(gold)
        pred_norm = normalize(pred)
        if not pred_norm:
            return OFF_TABLE
        if gold_norm == pred_norm:
            return EXACT
        if not self.contains(pred_norm):
            return OFF_TABLE
        if pred_norm in gold_norm:
            return HYPERNYM      # 模型答的是金标的上位词：答粗了
        if gold_norm in pred_norm:
            return HYPONYM       # 模型答的是金标的下位词：答细了
        return OTHER

    def is_confusable(self, a: object, b: object) -> bool:
        """两个类别是否互为易混（包含关系，或等长一字之差）。

        用来把「错在易混组内」和「错得毫无关系」分开报：前者补的是细粒度区分的
        数据，后者是模型压根没认出来。
        """
        left, right = normalize(a), normalize(b)
        if len(left) < 2 or len(right) < 2 or left == right:
            return False
        return left in right or right in left or one_char_apart(left, right)


def load_class_table(path: str | Path) -> ClassTable:
    """读 ``classes.yaml``（构建端那份 347 类的表）或等价的 JSON。

    ``names`` 既可以是 ``{0: "人员"}`` 映射，也可以是 ``["人员", ...]`` 列表，口径
    与构建端 ``load_class_table`` 一致。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到类别表文件：{path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - 环境问题，不是逻辑分支
            raise ImportError("读 yaml 类别表需要 PyYAML；或者把类别表转成 .json") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict) or data.get("names") is None:
        raise ValueError(f"{path} 里没有 names 字段")
    names = data["names"]
    if isinstance(names, dict):
        id2name = {int(k): str(v) for k, v in names.items()}
    elif isinstance(names, list):
        id2name = {i: str(v) for i, v in enumerate(names)}
    else:
        raise ValueError(f"names 字段格式无法识别：{type(names)}")
    declared = data.get("nc")
    if declared is not None and int(declared) != len(id2name):
        raise ValueError(f"{path} 里 nc={declared}，实际解析出 {len(id2name)} 个类别")
    return ClassTable(id2name=id2name)


def table_from_names(names: Iterable[str]) -> ClassTable:
    """从一串类别名直接建表，给测试和「没有类别表文件」的场景用。"""
    return ClassTable(id2name={i: str(n) for i, n in enumerate(names)})
