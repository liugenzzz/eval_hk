"""D 组描述的**代码前置检查**：范围合规、CHAIR 幻觉、空话率。纯代码，不调任何模型。

这三个数是 D 组唯一不受裁判偏置影响的客观锚（§15.3）。裁判是 Qwen3.8-27B，被测是
Qwen3-VL-8B，同家族 —— 同家族裁判可能偏爱同家族的输出风格，而我们没有异家族裁判
可以做自偏检测。所以这三个代码指标必须与裁判分**并列报**：两者走向不一致时，
以代码指标为准。

**范围合规必须走代码**（§9.1）。通用的「描述准确性」rubric 会给跑题答案高分（说得
没错啊），裁判判不出「跑题」这件事。词表直接复用数据构建端 ``prompts/describe/*.txt``
里的 ``#! must-not:`` 行 —— 那是生成这批数据时就用的同一份约束，重写一遍必然对不上。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .classes import ClassTable

# 用 [ \t]* 而不是 \s*：\s 会吃掉换行，于是空的 "#! must-not:" 会把下一行整行
# 抓成词表，contrast 和 full 这两个本来没有禁用词的 kind 会凭空多出一堆。
_KIND_LINE = re.compile(r"^#![ \t]*kind:[ \t]*(.+)$", re.MULTILINE)
_MUST_NOT_LINE = re.compile(r"^#![ \t]*must-not:[ \t]*(.*)$", re.MULTILINE)

# 「一辆车」「一个目标」这种什么都没说的答案。信息量维度由裁判打，但空话是能用代码
# 抓的那一部分 —— 它同样不受裁判偏置影响。
_FILLER_PATTERN = re.compile(r"^[一二三四五六七八九十\d]*\s*[个只名辆台艘架条把根]?\s*$")


@dataclass(frozen=True)
class ScopeRule:
    kind: str
    must_not: tuple[str, ...]

    def violations(self, text: object) -> tuple[str, ...]:
        haystack = str(text or "")
        return tuple(word for word in self.must_not if word and word in haystack)


def parse_scope_rule(text: str) -> ScopeRule | None:
    """从一个 describe 提示词文件里读出 kind 和 must-not 词表。"""
    kind_match = _KIND_LINE.search(text)
    if not kind_match:
        return None
    must_not_match = _MUST_NOT_LINE.search(text)
    words = must_not_match.group(1).split() if must_not_match else []
    return ScopeRule(kind=kind_match.group(1).strip(), must_not=tuple(words))


def load_scope_rules(prompt_dir: str | Path) -> dict[str, ScopeRule]:
    """读构建端的 ``prompts/describe/`` 目录，得到 kind -> 词表。"""
    directory = Path(prompt_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"describe 提示词目录不存在：{directory}")
    rules: dict[str, ScopeRule] = {}
    for path in sorted(directory.glob("*.txt")):
        rule = parse_scope_rule(path.read_text(encoding="utf-8"))
        if rule:
            rules[rule.kind] = rule
    if not rules:
        raise ValueError(f"{directory} 里没有带 '#! kind:' 的 describe 提示词")
    return rules


def rules_from_mapping(raw: Mapping[str, Iterable[str]]) -> dict[str, ScopeRule]:
    """配置里直接写词表的写法，给拿不到构建端目录的场景用。"""
    return {
        str(kind): ScopeRule(kind=str(kind), must_not=tuple(str(w) for w in words))
        for kind, words in raw.items()
    }


@dataclass(frozen=True)
class ChairResult:
    """CHAIR 幻觉（§5.2）：描述里提到的类别，有多少不在这张图的真值类别集合里。

    ``available=False`` 表示这张图**没有权威的类别集合**，此时不出数。用不完整的
    集合算 CHAIR 会系统性高估幻觉 —— 图里真实存在但没进评估集标注的目标，会被
    一个不落地记成模型编的。宁可标「无法计算」也不要给一个偏的数。
    """

    mentioned: tuple[str, ...]
    hallucinated: tuple[str, ...]
    available: bool

    @property
    def chair_i(self) -> float | None:
        """实例级：提到的类别里有多少是编的。"""
        if not self.available or not self.mentioned:
            return None
        return len(self.hallucinated) / len(self.mentioned)

    @property
    def chair_s(self) -> int | None:
        """句子级：这条描述里有没有出现过幻觉（0/1）。"""
        if not self.available:
            return None
        return int(bool(self.hallucinated))


def chair(text: object, gt_classes: Sequence[str] | None, table: ClassTable) -> ChairResult:
    mentioned = table.find_all_in_text(text)
    if gt_classes is None:
        return ChairResult(mentioned=mentioned, hallucinated=(), available=False)
    truth = {str(name).strip() for name in gt_classes if str(name).strip()}
    hallucinated = tuple(name for name in mentioned if name not in truth)
    return ChairResult(mentioned=mentioned, hallucinated=hallucinated, available=True)


def is_filler(text: object, label: object = "", min_chars: int = 6) -> bool:
    """空话：把类别名去掉之后基本什么都不剩（「一辆三轮车」「一个目标」）。

    这是「信息量」维度里能用代码抓的那一半，不受裁判偏置影响，与裁判的信息量分
    并列报。
    """
    raw = str(text or "").strip()
    if not raw:
        return True
    stripped = raw.replace(str(label or ""), "")
    stripped = re.sub(r"[\s，。、；：！？,.;:!?的了是有个]", "", stripped)
    if len(stripped) < min_chars:
        return True
    return bool(_FILLER_PATTERN.match(stripped))
