"""D 组描述 rubric 的解析。

**刻意不复用** ``judge_rubrics.parse_pointwise``：那一支是装备评估四个 rubric 版本
（v1/v2/v3/v4 加两个已退役格式）的解析中枢，它要保证历史数据按当初的口径读回来。
往里加一个新格式，等于让装备的历史回读多一条分支 —— 收益是省几十行代码，代价是
动了一个正在服役的、跨版本兼容的解析器。这里独立一份。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

RUBRIC_TAG = "describe_v1"
DIMENSIONS = ("correct", "grounded", "informative")
SCALE_MAX = 5

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class DescribeVerdict:
    correct: int
    grounded: int
    informative: int
    reason: str

    @property
    def mean_normalized(self) -> float:
        """三维平均后归一到 0~1。报表的 hit 列读它。

        三个维度**同时**分列报出来 —— 平均分只是给汇总用的一个句柄，
        「正确但空洞」和「具体但在编」平均下来可以是同一个数。
        """
        return (self.correct + self.grounded + self.informative) / (3.0 * SCALE_MAX)

    def as_columns(self) -> dict[str, Any]:
        return {
            "judge_correct": self.correct,
            "judge_grounded": self.grounded,
            "judge_informative": self.informative,
            "judge_reason": self.reason,
            "hit": round(self.mean_normalized, 4),
        }


def extract_json(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    match = _OBJECT.search(text)
    if not match:
        raise ValueError(f"裁判没有输出 JSON：{text[:200]!r}")
    return json.loads(match.group(0))


def parse_describe(content: str) -> DescribeVerdict:
    data = extract_json(content)
    scores: dict[str, int] = {}
    for dim in DIMENSIONS:
        if dim not in data:
            raise ValueError(f"裁判输出缺少维度 {dim}：{str(data)[:200]}")
        value = data[dim]
        try:
            score = int(round(float(value)))
        except (TypeError, ValueError):
            raise ValueError(f"维度 {dim} 不是数字：{value!r}") from None
        if not 1 <= score <= SCALE_MAX:
            raise ValueError(f"维度 {dim} 超出 1~{SCALE_MAX}：{score}")
        scores[dim] = score
    return DescribeVerdict(
        correct=scores["correct"],
        grounded=scores["grounded"],
        informative=scores["informative"],
        reason=str(data.get("reason", "") or ""),
    )
