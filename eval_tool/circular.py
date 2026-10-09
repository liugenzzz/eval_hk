"""CircularEval：同一道选择题把选项轮转 N 遍，每一遍都答对才算这道题对。

MMBench / VLMEvalKit 的做法。单次作答的准确率里混着位置偏好：模型偏爱某个字母、
或者正确答案恰好落在它习惯选的位置上。轮转之后这部分运气被剥掉 —— 四选一瞎猜，
单次能蒙对 25%，四遍全对只剩 (1/4)^4 ≈ 0.4%。

评估集侧：``expand_circular`` 把一条选择题展开成 N 条，``id`` 加 ``__r<k>`` 后缀，
``metadata.circular_id`` 指回原题，``metadata.rotation`` 记第几轮（0 = 原题）。
打分侧：``choice_circular`` 逐条抽字母，再按 ``circular_id`` 聚合（scorers/choice.py）。
"""

from __future__ import annotations

import copy
import re
from typing import Any

OPTION_LINE = re.compile(r"^[ \t]*([A-H])[ \t]*[.．、:：)）][ \t]*(.+?)[ \t]*$", re.M)
LETTERS = "ABCDEFGH"
_LEADING_LETTER = re.compile(r"^\s*[\(（\[]?\s*([A-H])(?![A-Za-z])")


def _options(question: str) -> list[re.Match[str]]:
    matches = list(OPTION_LINE.finditer(question))
    letters = "".join(m.group(1) for m in matches)
    if len(matches) < 2 or letters != LETTERS[: len(matches)]:
        raise ValueError(f"选项行必须是从 A 开始连续的 'A. …' 形式，读到的是 {letters or '（无）'}")
    return matches


def answer_letter(answer: str) -> str:
    match = _LEADING_LETTER.match(str(answer or ""))
    if not match:
        raise ValueError(f"答案不是以选项字母开头：{answer!r}")
    return match.group(1)


def rotate(question: str, answer: str, k: int) -> tuple[str, str, str]:
    """把选项内容整体后移 k 位。返回 (新题面, 新答案字母, 新答案选项文字)。

    第 i 个位置放原来第 (i-k) 个选项，所以原答案从位置 a 挪到 (a+k) mod n。
    """
    matches = _options(question)
    n = len(matches)
    texts = [m.group(2) for m in matches]
    rotated = [texts[(i - k) % n] for i in range(n)]
    out = question
    for m, text in sorted(zip(matches, rotated), key=lambda pair: pair[0].start(2), reverse=True):
        out = out[: m.start(2)] + text + out[m.end(2):]
    new_index = (LETTERS.index(answer_letter(answer)) + k) % n
    return out, LETTERS[new_index], rotated[new_index]


def expand_circular(record: dict[str, Any], n: int | None = None) -> list[dict[str, Any]]:
    """一条 ShareGPT 单轮选择题 → n 条轮转题（默认 n = 选项数）。"""
    conv = record.get("conversations") or []
    if len(conv) != 2 or conv[0].get("from") != "human" or conv[1].get("from") != "gpt":
        raise ValueError(f"{record.get('id')}: CircularEval 只接受单轮问答")
    question, answer = conv[0]["value"], conv[1]["value"]
    options = len(_options(question))
    n = options if n is None else n
    with_text = bool(re.match(r"^\s*[A-H]\s*[.．、:：)）]\s*\S", answer))
    out = []
    for k in range(n):
        q, letter, text = rotate(question, answer, k)
        item = copy.deepcopy(record)
        item["id"] = f"{record['id']}__r{k}"
        item["conversations"][0]["value"] = q
        item["conversations"][1]["value"] = f"{letter}. {text}" if with_text else letter
        meta = item.setdefault("metadata", {})
        meta.update({"circular_id": str(record["id"]), "rotation": k, "circular_n": n})
        out.append(item)
    return out
