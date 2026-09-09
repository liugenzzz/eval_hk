"""C / E 组的同义兜底：模型自创了一个类别表里没有的词时，问裁判一次。

§6：「答案不在类别表里（模型自创了词）才丢给裁判判一次是不是同义 —— 这是 C / E 组
唯一用到裁判的地方。」代码判得了的绝不问裁判：一是省钱，二是代码判的结果重跑一百遍
逐位相同，而裁判会抖。

判词独立解析，**不碰** ``judge_rubrics.parse_pointwise`` —— 那一支是装备评估四个
rubric 版本的解析中枢，要保证历史数据按当初口径读回来。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

RUBRIC_TAG = "synonym_v1"
EXACT, HYPERNYM, HYPONYM, NONE = "exact", "hypernym", "hyponym", "none"
RELATIONS = (EXACT, HYPERNYM, HYPONYM, NONE)

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class SynonymVerdict:
    same: bool
    relation: str
    reason: str

    def as_columns(self) -> dict[str, Any]:
        return {
            "judge_same": int(self.same),
            "judge_relation": self.relation,
            "judge_reason": self.reason,
        }


def parse_synonym(content: str) -> SynonymVerdict:
    text = str(content or "").strip()
    fence = _FENCE.search(text)
    if fence:
        text = fence.group(1).strip()
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        match = _OBJECT.search(text)
        if not match:
            raise ValueError(f"裁判没有输出 JSON：{text[:200]!r}") from None
        data = json.loads(match.group(0))
    if "same" not in data:
        raise ValueError(f"裁判输出缺少 same：{str(data)[:200]}")
    same = bool(data["same"])
    relation = str(data.get("relation", EXACT if same else NONE)).strip().lower()
    if relation not in RELATIONS:
        raise ValueError(f"relation 取值非法：{relation!r}")
    # same=false 时 relation 必须是 none，否则两个字段自相矛盾，判词不可信。
    if not same and relation != NONE:
        raise ValueError(f"same=false 但 relation={relation!r}，判词自相矛盾")
    if same and relation == NONE:
        raise ValueError("same=true 但 relation=none，判词自相矛盾")
    return SynonymVerdict(same=same, relation=relation, reason=str(data.get("reason", "") or ""))


def ask(client: Any, prompt: str, gold: str, prediction: str, retries: int = 2) -> dict[str, Any]:
    """问一次裁判。判不出来时返回 same=NA，**不记 0** —— 判失败和判「不是同义」
    是两件事，记 0 会在两个模型的自创词比例不同时把对比拉出方向性偏差。"""
    user_text = (
        f"金标类别：{gold}\n"
        f"被测模型的答案：{prediction}\n\n"
        "这两个说法是不是指同一类东西？只输出 JSON。"
    )
    last_error = ""
    for _ in range(max(1, retries)):
        try:
            return parse_synonym(client.judge_raw(prompt, user_text)).as_columns()
        except Exception as exc:  # noqa: BLE001 - 判词千奇百怪，原样带回报表
            last_error = f"{type(exc).__name__}: {exc}"
    return {"judge_same": None, "judge_relation": "", "judge_reason": f"[judge_error] {last_error}"}
