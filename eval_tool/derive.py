"""派生评估集：把「模型自己的输出」折回成新的评估集，再当普通数据集跑一遍。

为什么这么做，而不是让推理引擎支持 history_mode / 反向问 / 问法扰动：

装备和书籍走的是同一条 ``run_infer``。那里面有 manifest 指纹、``_partial/*.jsonl``
分片、``InferenceLock``、并行 worker 的索引合并。改坏了不是报错，是几小时的 GPU 跑到
一半续不上，或者更糟 —— 续上了但接错了行。

派生法把这三件事全部变成「生成一份新的 jsonl，然后当普通数据集跑」：

    ① infer(ground_box)              现有通路，一行不改
    ② derive -> describe_modelhist.jsonl   纯函数，不需要 GPU，可单测
    ③ infer(describe_modelhist)      现有通路，一行不改，自己的指纹/续传/缓存
    ④ 报表层比 ② 和主线，得到链路衰减率

代价是磁盘上多几份派生集；换来的是推理引擎一个字符都不用动，而且每份派生集的断点
续传、缓存、重跑全部免费继承。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .bbox import iou, parse_boxes
from .eval_set import split_turns
from .io import load_prediction_file
from .phrase_pool import PhrasePool, format_bbox

MODEL_HISTORY_SUFFIX = "__mh"
REVERSE_SUFFIX = "__rev"
PERTURB_SUFFIX = "__pt"

TASK_REVERSE = "reverse_consistency"
TASK_PERTURB = "question_perturbation"


def load_predictions(paths: Iterable[str | Path]) -> dict[str, str]:
    """把若干份预测文件并成 index -> 预测文本。

    一条 inventory_locate 的三轮分散在三个数据集里（清单 / 单框 / 描述），拼模型历史
    需要前两轮的预测，所以这里接受多份文件。同一个 index 在两份文件里出现是配置错了，
    直接报错 —— 静默取后一份会让派生集里混进另一次跑的结果。
    """
    merged: dict[str, str] = {}
    for path in paths:
        frame = load_prediction_file(path)
        for _, row in frame.iterrows():
            index = str(row["index"])
            prediction = row.get("prediction", "")
            text = "" if prediction is None or pd.isna(prediction) else str(prediction)
            if index in merged and merged[index] != text:
                raise ValueError(f"预测文件之间 index 冲突：{index}（{path}）")
            merged[index] = text
    return merged


def _keep(sample_id: str, ratio: float, salt: str) -> bool:
    """按 id 哈希做确定性抽样。

    用哈希而不是 random.sample：评估集加了一条样本，或者记录顺序变了，抽中的那一批
    也不该跟着变 —— 否则两个 checkpoint 的链路衰减率算在不同的子集上，没法比。
    """
    if ratio >= 1.0:
        return True
    if ratio <= 0.0:
        return False
    digest = hashlib.sha256(f"{salt}|{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64) < ratio


def _turn_index(record_id: str, turn: int) -> str:
    return f"{record_id}__t{turn}"


def derive_model_history(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, str],
    *,
    target_turns: Mapping[str, int] | None = None,
    sample_ratio: float = 1.0,
    salt: str = "model_history",
) -> list[dict[str, Any]]:
    """§8：把历史轮的标准答案换成**模型自己的**输出，只保留到目标轮。

    第 1 轮只跑一次，gold 历史和 model 历史共用它的结果。链路衰减率是统计量，
    不需要全量 —— ``sample_ratio`` 默认给 0.3 那一档用。

    目标轮的标准答案保持不变：换的是**上文**，不是真值。
    """
    target_turns = target_turns or {}
    out: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("id", ""))
        metadata = dict(record.get("metadata") or {})
        task = str(metadata.get("task_type", "") or "")
        turns = split_turns(record.get("conversations") or [])
        if len(turns) < 2:
            continue
        target = int(target_turns.get(task, len(turns)))
        if not 2 <= target <= len(turns):
            continue
        if not _keep(record_id, sample_ratio, salt):
            continue

        needed = [_turn_index(record_id, turn) for turn in range(1, target)]
        if any(index not in predictions for index in needed):
            # 缺哪一轮的预测就整条跳过。用 gold 补一半、model 补一半算出来的衰减率，
            # 分子分母不是同一件事。
            continue

        conversations: list[dict[str, str]] = []
        for turn, (human, gold) in enumerate(turns[:target], start=1):
            conversations.append({"from": "human", "value": human})
            if turn < target:
                conversations.append({"from": "gpt", "value": predictions[_turn_index(record_id, turn)]})
            else:
                conversations.append({"from": "gpt", "value": gold})

        metadata.update({
            "history_mode": "model",
            "derived_from": record_id,
            "derived_kind": "model_history",
            "target_turn": target,
            "n_turns": target,
        })
        out.append({
            "id": f"{record_id}{MODEL_HISTORY_SUFFIX}",
            "images": list(record.get("images") or record.get("image") or []),
            "conversations": conversations,
            "metadata": metadata,
        })
    return out


def derive_reverse_consistency(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, str],
    *,
    question_pool: PhrasePool,
    answer_template: str = "该区域内的是{label}。",
    box_turn: int = 1,
    scale: int = 1000,
    tasks: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """§11.1 双向一致性（借 Ferret 的 Referring ↔ Grounding 对偶）。

    正向问「框出白色面包车」得到框 B，反向拿 B 问「这个框里是什么」，看答出来的类别
    和「白色面包车」对不对得上。**不需要任何额外标注** —— 数据集本来就同时有
    ``ground_*``（文字→框）和 ``region_identify``（框→文字）。

    它测的是模型有没有真的建立文字 ↔ 区域的**双向映射**，还是只单向背下了
    「看到『框出』就吐坐标」。

    **不按正向对错筛样本。** 正向框错时反向答「卡车」对那个框来说是对的，但和原来的
    指代对不上 —— 那恰恰是要测的失败模式。正向的 IoU 照样记进 metadata，报表想拆
    「两边都对」和「反向一致但正向框错」的时候有依据。
    """
    allowed = set(tasks) if tasks else None
    out: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("id", ""))
        metadata = dict(record.get("metadata") or {})
        task = str(metadata.get("task_type", "") or "")
        if allowed is not None and task not in allowed:
            continue
        label = str(metadata.get("label", "") or "")
        if not label:
            continue
        turns = split_turns(record.get("conversations") or [])
        if len(turns) < box_turn:
            continue
        prediction = predictions.get(_turn_index(record_id, box_turn))
        if prediction is None:
            continue
        predicted = parse_boxes(prediction, scale=scale)
        if not predicted.ok:
            # 正向根本没框出来，反向问无从问起。这一条不进双向一致率的分母 ——
            # 它已经被正向的格式合规率记过一次了，再记一次是重复惩罚。
            continue
        box = predicted.boxes[0]
        if box.area <= 0:
            # 框整个飞出画面时（x1、x2 都 > scale），裁剪后两边都贴到边界，退化成
            # 零宽或零高。拿它去问「这个区域里是什么」问的是一块零面积的地方，模型
            # 答什么都没有意义，而这条样本在正向已经被记过一次错了。同上，丢掉。
            continue
        gold = parse_boxes(turns[box_turn - 1][1], scale=scale)
        forward_iou = iou(box, gold.boxes[0]) if gold.ok else None

        question = question_pool.render({"bbox": format_bbox(box.as_tuple())}, 1, record_id)
        if not question:
            continue

        metadata.update({
            "task_type": TASK_REVERSE,
            "derived_from": record_id,
            "derived_kind": "reverse_consistency",
            "forward_task_type": task,
            "forward_iou": None if forward_iou is None else round(float(forward_iou), 4),
            "forward_box": list(box.as_tuple()),
            "n_turns": 1,
        })
        out.append({
            "id": f"{record_id}{REVERSE_SUFFIX}",
            "images": list(record.get("images") or record.get("image") or []),
            "conversations": [
                {"from": "human", "value": "<image>\n" + question[0]},
                {"from": "gpt", "value": answer_template.replace("{label}", label)},
            ],
            "metadata": metadata,
        })
    return out


def derive_question_perturbation(
    records: Sequence[Mapping[str, Any]],
    *,
    pools: Mapping[str, PhrasePool],
    variants: int = 3,
    box_turn: int = 1,
    sample_ratio: float = 1.0,
    salt: str = "perturbation",
    tasks: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """§11.2 问法扰动：同一个目标用问法池里 ``variants`` 种不同说法问。

    答案框应该一样。报三次输出之间的四点偏差方差和 IoU 一致率 —— 这测的是对问法的
    过拟合，而标准评测集完全测不出来（它们的问法也是固定的）。

    每个变体是一条独立的单轮记录，带同一个 ``perturb_group``；打分器按组算一致性。
    """
    allowed = set(tasks) if tasks else None
    out: list[dict[str, Any]] = []
    for record in records:
        record_id = str(record.get("id", ""))
        metadata = dict(record.get("metadata") or {})
        task = str(metadata.get("task_type", "") or "")
        if allowed is not None and task not in allowed:
            continue
        if not _keep(record_id, sample_ratio, salt):
            continue
        pool = pools.get(task) or pools.get("default")
        if pool is None:
            continue
        turns = split_turns(record.get("conversations") or [])
        if len(turns) < box_turn:
            continue
        original_question = turns[box_turn - 1][0].replace("<image>", "").strip()
        values = {
            "label": str(metadata.get("label", "") or ""),
            "attribute": str(metadata.get("attribute", "") or ""),
        }
        questions = pool.render(values, variants, record_id, exclude=[original_question])
        if len(questions) < variants:
            # 凑不满就整条不做：两个变体和三个变体算出来的方差不是同一个量。
            continue
        for position, question in enumerate(questions):
            variant_meta = dict(metadata)
            variant_meta.update({
                "task_type": TASK_PERTURB,
                "derived_from": record_id,
                "derived_kind": "question_perturbation",
                "forward_task_type": task,
                "perturb_group": record_id,
                "perturb_variant": position,
                "n_turns": 1,
            })
            out.append({
                "id": f"{record_id}{PERTURB_SUFFIX}{position}",
                "images": list(record.get("images") or record.get("image") or []),
                "conversations": [
                    {"from": "human", "value": "<image>\n" + question},
                    {"from": "gpt", "value": turns[box_turn - 1][1]},
                ],
                "metadata": variant_meta,
            })
    return out


def write_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out
