"""评估集（``test.jsonl``）直读：按轮次拆行、metadata 扁平化、按任务/轮次选取。

数据构建端产出的 ``test.jsonl`` 恒带 metadata（``output.include_metadata`` 只作用于
train/val）。评估报表要按 ``task_type`` / ``difficulty`` / ``size_bucket`` / ``describe_kind``
/ 数量档拆开看，摘掉 metadata 就拆不了，所以这条通路直接读 jsonl，不经过 TSV 转换。

**多轮任务按轮次分别归组**，不是整条归一组。一条 ``inventory_locate`` 样本三轮各答
一种东西：轮 1 是文本清单、轮 2 只给一个框、轮 3 是描述 —— 分别落在计数、单框、
描述三个组，用三个不同的打分器。所以一条记录在这里会摊成多行，每行带 ``turn``，
数据集配置用 ``select`` 声明自己要哪些任务的哪一轮::

    "ground_box": {
      "name": "eval_set_v1", "kind": "grounding_single",
      "params": {"select": [
        {"task_type": ["ground_appearance", "ground_full", "ground_relation"], "turn": 1},
        {"task_type": ["inventory_locate"], "turn": 2}
      ]}
    }

``turn`` 从 **1** 开始，和需求文档里「轮 1 / 轮 2 / 轮 3」的说法一致（注意与
``convert_vqa_json`` 的 ``__t0`` 后缀不同，那是另一条通路的历史约定）。

历史轮按 **gold 回放**塞进 ``history`` 列，形状与 ``convert_vqa_json`` 一致，推理端
不用改。``history_mode: model`` 是另一件事，在阶段 5。

对话字段认两种形状：ShareGPT 的 ``conversations``（``from``/``value``）和 OpenAI 的
``messages``（``role``/``content``，content 可以是字符串或多模态数组）。书籍那批评估集
是后一种。两者都不带的记录（例如只有 ``text`` 的语料行）产不出问答对，直接跳过并计数 ——
没有问题也没有参考答案，裁判无从打分。

``category`` 列默认放 ``metadata.task_type``（目标检测按任务拆报表）。书籍那种每条自带
领域分类的评估集，用 ``params.category_field`` 指到那个字段上，报表维度就按它拆。
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .imaging import encode_image_cell

META_PREFIX = "meta."

DEFAULT_CATEGORY = "未分类"
"""``category_field`` 指定的字段缺失或为空时的兜底分类。"""

# OpenAI ``messages`` 的 role 映射到本模块其余部分说的 ShareGPT ``from``。
# 表外的角色（system / tool / ...）丢弃：它们既没有问题也没有参考答案。
_ROLE_TO_FROM = {
    "human": "human",
    "user": "human",
    "gpt": "gpt",
    "assistant": "gpt",
    "model": "gpt",
}


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """读 jsonl（每行一条）或 json 数组，按文件内容判断，不看扩展名。"""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError(f"{path} 不是记录数组")
        return data
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: 这一行不是合法 JSON（{exc}）") from exc
    return records


def sha256_of(path: str | Path) -> str:
    """评估集内容哈希。§13 要求评估集冻结：不同 checkpoint 之间可比的前提是
    评的是同一批样本，指纹对不上的结果不许画进同一张图。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_turns(conversations: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(conversations):
        human, gpt = conversations[i], conversations[i + 1]
        if human.get("from") != "human" or gpt.get("from") != "gpt":
            i += 1
            continue
        turns.append((str(human.get("value", "")), str(gpt.get("value", ""))))
        i += 2
    return turns


def clean_question(value: str) -> str:
    return str(value or "").replace("<image>", "").strip()


def content_to_text(content: Any) -> str:
    """把一条消息的正文压成 ShareGPT 的 ``value`` 字符串。

    数组正文是 OpenAI 多模态 content，每个图片片段折算一个 ``<image>`` 标记，这样
    ``record_to_rows`` 里按标记数分配图片的逻辑一个字都不用改。
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "").strip().lower()
        if kind in {"image", "image_url"} or "image_url" in item or "image" in item:
            parts.append("<image>")
            continue
        text = item.get("text")
        if text is None:
            text = item.get("value")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def normalize_conversations(record: Mapping[str, Any]) -> list[dict[str, str]]:
    """取出记录的对话，统一成 ShareGPT 的 ``{"from", "value"}``。

    两种都没有就返回空表 —— 只有 ``text`` 的纯语料行会走到这里。
    """
    conversations = record.get("conversations")
    if isinstance(conversations, list) and conversations:
        return [
            {"from": str(turn.get("from", "")), "value": content_to_text(turn.get("value"))}
            for turn in conversations
            if isinstance(turn, Mapping)
        ]
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = str(message.get("role") or message.get("from") or "").strip().lower()
            mapped = _ROLE_TO_FROM.get(role)
            if mapped is None:
                continue
            body = message.get("content")
            if body is None:
                body = message.get("value")
            normalized.append({"from": mapped, "value": content_to_text(body)})
        return normalized
    return []


def selection_matches(row: Mapping[str, Any], select: Sequence[Mapping[str, Any]] | None) -> bool:
    """``select`` 是若干条 {task_type: [...], turn: n} 的或关系；留空表示全要。"""
    if not select:
        return True
    for rule in select:
        tasks = rule.get("task_type")
        if tasks is not None:
            wanted = [tasks] if isinstance(tasks, str) else list(tasks)
            if str(row.get("task_type", "")) not in {str(t) for t in wanted}:
                continue
        turn = rule.get("turn")
        if turn is not None:
            turns = [turn] if isinstance(turn, int) else list(turn)
            if int(row.get("turn", 0)) not in {int(t) for t in turns}:
                continue
        return True
    return False


def _flatten_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """metadata 扁平化成 ``meta.*`` 列。嵌套的值（inventory 那种列表）保持原样，
    解析交给用它的打分器 —— 在这里 json.dumps 一遍，读的人还得再解一次。"""
    return {f"{META_PREFIX}{key}": value for key, value in metadata.items()}


def category_of(
    record: Mapping[str, Any],
    metadata: Mapping[str, Any],
    category_field: str | None,
    default_category: str = DEFAULT_CATEGORY,
) -> str | None:
    """``category_field`` 指定的分类：先看记录顶层，再看 metadata，都没有才兜底。

    书籍那批把 ``category`` 写在记录顶层，目标检测那批什么都在 ``metadata`` 里，
    两种都认，配置只需要写字段名。返回 None 表示没开这个开关，保持原有语义。
    """
    if not category_field:
        return None
    value = record.get(category_field)
    if value is None or not str(value).strip():
        value = metadata.get(category_field)
    text = "" if value is None else str(value).strip()
    return text or default_category


def record_to_rows(
    record: Mapping[str, Any],
    *,
    image_root: Path | None = None,
    image_cache: dict[str, str] | None = None,
    category_field: str | None = None,
    default_category: str = DEFAULT_CATEGORY,
) -> list[dict[str, Any]]:
    record_id = str(record.get("id", ""))
    metadata = record.get("metadata") or {}
    task_type = str(metadata.get("task_type", "") or "")
    image_names = record.get("images") or record.get("image") or []
    if isinstance(image_names, str):
        image_names = [image_names]
    turns = split_turns(normalize_conversations(record))
    category = category_of(record, metadata, category_field, default_category)

    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    images_used = 0
    for turn_index, (human_value, gpt_value) in enumerate(turns, start=1):
        tag_count = human_value.count("<image>")
        images_used += tag_count
        cumulative = image_names[:images_used] if images_used else image_names
        question = clean_question(human_value)
        answer = str(gpt_value).strip()
        rows.append(
            {
                "index": f"{record_id}__t{turn_index}",
                "sample_id": record_id,
                "task_type": task_type,
                "turn": turn_index,
                "n_turns": len(turns),
                "image": _encode_images(cumulative, image_root, image_cache),
                "image_files": ",".join(str(name) for name in cumulative),
                "question": question,
                "answer": answer,
                "history": json.dumps(history, ensure_ascii=False) if history else "",
                # category / l2-category 是现有报表层的两根默认分组轴，直接挂上
                # task_type 和难度档，不用等报表重写就能拆开看。
                # 开了 category_field（书籍那种每条自带领域分类）就换成那个分类，
                # task_type 仍单独成列，两根轴都还在。
                "category": task_type if category is None else category,
                "l2-category": str(metadata.get("difficulty") or ""),
                "source_id": str(metadata.get("source_image") or record_id),
                **_flatten_metadata(metadata),
            }
        )
        history.append({"q": question, "a": answer, "n_img": tag_count})
    return rows


def _encode_images(
    names: Iterable[Any], image_root: Path | None, cache: dict[str, str] | None
) -> str:
    """图片编成 base64 塞进 image 列（沿用现有通路的约定）。

    没给 image_root 就留空 —— 代码打分器（画框、计数、识别）压根不看图，为了跑一次
    坐标打分把几个 G 的图读进内存没有道理。裁判打分和推理才需要图。
    """
    if image_root is None:
        return ""
    cache = cache if cache is not None else {}
    encoded: list[str] = []
    for name in names:
        key = str(name)
        if key not in cache:
            path = Path(image_root) / key
            cache[key] = base64.b64encode(path.read_bytes()).decode("ascii")
        encoded.append(cache[key])
    return encode_image_cell(encoded) if encoded else ""


def load_eval_set(
    path: str | Path,
    *,
    select: Sequence[Mapping[str, Any]] | None = None,
    image_root: str | Path | None = None,
    category_field: str | None = None,
    default_category: str = DEFAULT_CATEGORY,
) -> pd.DataFrame:
    records = load_records(path)
    cache: dict[str, str] = {}
    root = Path(image_root) if image_root else None
    rows: list[dict[str, Any]] = []
    text_only = 0
    no_turns = 0
    for record in records:
        record_rows = record_to_rows(
            record,
            image_root=root,
            image_cache=cache,
            category_field=category_field,
            default_category=default_category,
        )
        if not record_rows:
            # 产不出问答对的记录分两种，报表上都是「少了几条」，但原因完全不同：
            # 纯语料行是数据本来就这样，没有问答轮次多半是格式写错了。
            if _is_text_only(record):
                text_only += 1
            else:
                no_turns += 1
            continue
        for row in record_rows:
            if selection_matches(row, select):
                rows.append(row)
    if text_only:
        print(f"[eval_set] {path}: 跳过 {text_only} 条只有 text 字段的记录"
              f"（没有问题/参考答案，无法判分）", flush=True)
    if no_turns:
        print(f"[eval_set] {path}: 跳过 {no_turns} 条没有可用问答轮次的记录", flush=True)
    if not rows:
        # 选空了是配置写错了（任务名拼错、轮次填反），静默返回空表会让报表里多一格
        # 「样本不足」，而那格实际上是 bug。
        raise ValueError(f"{path}: select 没有选中任何样本，检查 task_type 和 turn")
    frame = pd.DataFrame(rows)
    frame["index"] = frame["index"].astype(str)
    return frame


def _is_text_only(record: Mapping[str, Any]) -> bool:
    """只有 ``text`` 的语料行，不是问答记录。"""
    return bool(record.get("text")) and not record.get("conversations") and not record.get("messages")


def write_tsv(frame: pd.DataFrame, out_path: str | Path) -> Path:
    """写成推理端认得的 TSV。推理走的是 TSV 通路，这一步把评估集喂给它。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, sep="\t", index=False, encoding="utf-8-sig")
    return out
