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


def record_to_rows(
    record: Mapping[str, Any],
    *,
    image_root: Path | None = None,
    image_cache: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    record_id = str(record.get("id", ""))
    metadata = record.get("metadata") or {}
    task_type = str(metadata.get("task_type", "") or "")
    image_names = record.get("images") or record.get("image") or []
    if isinstance(image_names, str):
        image_names = [image_names]
    turns = split_turns(record.get("conversations") or [])

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
                "category": task_type,
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
) -> pd.DataFrame:
    records = load_records(path)
    cache: dict[str, str] = {}
    root = Path(image_root) if image_root else None
    rows: list[dict[str, Any]] = []
    for record in records:
        for row in record_to_rows(record, image_root=root, image_cache=cache):
            if selection_matches(row, select):
                rows.append(row)
    if not rows:
        # 选空了是配置写错了（任务名拼错、轮次填反），静默返回空表会让报表里多一格
        # 「样本不足」，而那格实际上是 bug。
        raise ValueError(f"{path}: select 没有选中任何样本，检查 task_type 和 turn")
    frame = pd.DataFrame(rows)
    frame["index"] = frame["index"].astype(str)
    return frame


def write_tsv(frame: pd.DataFrame, out_path: str | Path) -> Path:
    """写成推理端认得的 TSV。推理走的是 TSV 通路，这一步把评估集喂给它。"""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, sep="\t", index=False, encoding="utf-8-sig")
    return out
