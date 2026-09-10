"""D 组自由描述：代码前置检查 + 裁判分维度打分。

分工写死（§9.1）：

============  ======  ==============================================
维度           引擎     说明
============  ======  ==============================================
正确性         裁判     说的内容图里有吗（幻觉）
落地性         裁判     说的是不是**这个框里**的东西，而不是图里别处的
信息量         裁判     是不是「一辆车」这种空话
**范围合规**   **代码**  appearance 有没有跑去说方位、position 有没有跑去说外观
============  ======  ==============================================

**范围合规必须走代码。** 通用的「描述准确性」rubric 会给跑题答案高分（说得没错啊），
裁判判不出「跑题」这件事。代码判更准、更省钱、可复现，词表直接复用构建端
``prompts/describe/*.txt`` 的 ``#! must-not:``。

另外两个代码指标 —— CHAIR 幻觉率和空话率 —— 与裁判分**并列报**（§15.3）。裁判是
Qwen3.8-27B，被测是 Qwen3-VL-8B，同家族，没有异家族裁判可以做自偏检测。这三个代码
指标完全不受裁判偏置影响，是 D 组唯一的客观锚：两者走向不一致时以代码指标为准。

关了 ``do_pointwise`` 时只出代码那三个数，一次裁判都不调 —— 那三个本来就是最先该看的。
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Mapping

import pandas as pd

from ..metrics_text import aux_metrics
from ..classes import ClassTable, load_class_table, table_from_names
from ..compliance import TEXT
from ..counting import parse_inventory_gold
from ..describe_rubric import DIMENSIONS, parse_describe
from ..describe_scope import ScopeRule, chair, is_filler, load_scope_rules, rules_from_mapping
from ..image_classes import ImageClassIndex, declared_image_classes
from ..prompting import load_prompt_text
from . import JUDGE, ScoringContext, register, resolve_path

DEFAULT_PROMPT = "prompts/judge_describe_v1.txt"


def _scope_rules(ctx: ScoringContext) -> dict[str, ScopeRule]:
    """词表来源：构建端的 describe 提示词目录，或配置里直接写。都没有就不判范围合规。"""
    inline = ctx.params.get("must_not")
    if inline:
        return rules_from_mapping(inline)
    directory = ctx.params.get("describe_prompt_dir")
    if directory:
        return load_scope_rules(resolve_path(ctx, directory))
    return {}


def _class_table(ctx: ScoringContext, data: pd.DataFrame) -> tuple[ClassTable | None, bool]:
    """返回 (类别表, 是不是权威的)。

    CHAIR 要两头都权威才算得出来：类别集合要完整（inventory），**类别表也要完整**。
    兜底表只含评估集里出现过的 label —— 模型编出一个「船」，表里没有「船」，
    这个词根本不会被识别成类别，幻觉就被漏报了。这个方向和类别集合不完整时的
    高估正好相反，两个都会让 CHAIR 变成一个偏的数。
    """
    configured = ctx.params.get("classes_yaml") or ctx.params.get("classes_path")
    if configured:
        return load_class_table(resolve_path(ctx, configured)), True
    label_col = next((c for c in ("meta.label", "label") if c in data.columns), None)
    if label_col is None:
        return None, False
    names = {str(v).strip() for v in data[label_col].dropna().tolist() if str(v).strip()}
    return (table_from_names(sorted(names)) if names else None), False


# describe 用自己的这一份而不是 object_ident 的：那边找不到类别表会抛，而 D 组没有
# 类别表照样能出范围合规和空话率两个指标，只是 CHAIR 不出。


def _inventory_class_sets(data: pd.DataFrame) -> dict[str, list[str]]:
    """从 ``meta.inventory`` 拿到的类别集合，按 source_image 索引。

    只有 ``inventory_locate`` 那一种样本带 inventory，所以这条路覆盖不到四分之一的
    描述样本 —— 它是最后的兜底，优先走原始标注文件那条（见 ``image_classes`` 模块）。
    """
    if "meta.inventory" not in data.columns:
        return {}
    source_col = next((c for c in ("meta.source_image", "source_id") if c in data.columns), None)
    if source_col is None:
        return {}
    out: dict[str, list[str]] = {}
    for _, row in data.iterrows():
        raw = row.get("meta.inventory")
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            continue
        inventory = parse_inventory_gold(raw)
        if inventory.ok:
            out[str(row.get(source_col, ""))] = sorted(inventory.labels)
    return out


def _gt_classes(
    row: Mapping[str, Any],
    index: ImageClassIndex,
    inventory_sets: Mapping[str, list[str]],
    source_col: str | None,
) -> tuple[str, ...] | None:
    """这张图的完整类别集合。三个来源按可信度排，取第一个拿得到的。"""
    declared = declared_image_classes(row)
    if declared is not None:
        return declared
    from_labels = index.classes_of(row.get("meta.source_annotation") or row.get("source_annotation"))
    if from_labels is not None:
        return from_labels
    if source_col:
        found = inventory_sets.get(str(row.get(source_col, "")))
        if found is not None:
            return tuple(found)
    return None


def _describe_kind(row: Mapping[str, Any]) -> str:
    for column in ("meta.describe_kind", "describe_kind"):
        value = row.get(column)
        if value is not None and not _is_na(value) and str(value).strip():
            return str(value).strip()
    return ""


def _upstream_form(row: Mapping[str, Any]) -> str:
    """§2.1：D 组内部按**上游形态**拆行报，不能混在一起平均。

    - 文字指代（7 个 ground_*）：听懂一句话说的是哪个目标
    - 坐标回指（detect_describe 轮 2）：把一串坐标对回到具体那个实例
    - 上文承接（inventory_locate 轮 3）：接住自己上一轮说过的话

    这是三种不同的能力，平均成一个数就看不出是哪一种垮了。
    """
    task = str(row.get("task_type", "") or "")
    if task == "detect_describe":
        return "坐标回指"
    if task == "inventory_locate":
        return "上文承接"
    if task.startswith("ground_"):
        return "文字指代"
    return ""


def _is_na(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


@register("describe", engine=JUDGE, pairwise=True, answer_form=TEXT)
def score_describe(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    rules = _scope_rules(ctx)
    table, table_is_authoritative = _class_table(ctx, data)
    inventory_sets = _inventory_class_sets(data)
    labels_dir = ctx.params.get("labels_dir")
    index = ImageClassIndex(
        resolve_path(ctx, labels_dir) if labels_dir else None,
        table if table_is_authoritative else None,
    )
    source_col = next((c for c in ("meta.source_image", "source_id") if c in data.columns), None)

    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        prediction = row.get("prediction", "")
        kind = _describe_kind(row)
        rule = rules.get(kind)
        violations = rule.violations(prediction) if rule else ()
        label = row.get("meta.label") or row.get("label") or ""

        record: dict[str, Any] = {
            "describe_kind": kind,
            "upstream_form": _upstream_form(row),
            # 范围合规：代码判，裁判判不出「跑题」。词表来自构建这批数据时的同一份约束。
            "scope_ok": pd.NA if rule is None else int(not violations),
            "scope_violations": ",".join(violations),
            # 空话率：信息量里能用代码抓的那一半，不受裁判偏置影响。
            "is_filler": int(is_filler(prediction, label)),
        }

        if table is None:
            record.update({"chair_i": math.nan, "chair_s": pd.NA, "chair_available": 0,
                           "mentioned_classes": "", "hallucinated_classes": ""})
        else:
            gt_classes = (
                _gt_classes(row, index, inventory_sets, source_col)
                if table_is_authoritative
                else None
            )
            result = chair(prediction, list(gt_classes) if gt_classes else gt_classes, table)
            record.update(
                {
                    "chair_i": result.chair_i if result.chair_i is not None else math.nan,
                    "chair_s": pd.NA if result.chair_s is None else result.chair_s,
                    "chair_available": int(result.available),
                    "mentioned_classes": ",".join(result.mentioned),
                    "hallucinated_classes": ",".join(result.hallucinated),
                }
            )
        rows.append(record)

    scored = data.copy()
    for column in rows[0] if rows else []:
        scored[column] = [row[column] for row in rows]

    # 文本重合度四列 judge_text 一直有，describe 漏了 —— 而 describe 也是
    # pairwise=True，成对判定要读 pred_len，缺了它整趟评估会在最后一步崩掉。
    aux = [aux_metrics(row.get("answer", ""), row.get("prediction", ""))
           for _, row in scored.iterrows()]
    for column in ("bleu1", "bleu2", "rouge_l", "pred_len"):
        scored[column] = [item[column] for item in aux]

    if not ctx.do_pointwise:
        # 代码那三个数一次裁判都不调就能出，它们本来就是最先该看的。
        scored["hit"] = pd.NA
        for dim in DIMENSIONS:
            scored[f"judge_{dim}"] = pd.NA
        scored["judge_reason"] = "[pointwise_disabled]"
        return scored

    return _judge(scored, ctx)


def _judge(scored: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    prompt = _prompt_text(ctx)
    cache = ctx.judge_cache
    client = ctx.judge_client
    fingerprint = getattr(getattr(client, "settings", None), "fingerprint", "")
    image_map = dict(ctx.image_map)

    rows = scored.to_dict("records")
    results: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, dict[str, Any]]] = []
    for position, row in enumerate(rows):
        index = str(row.get("index", position))
        prediction = row.get("prediction", "")
        if prediction is None or _is_na(prediction) or not str(prediction).strip():
            results[index] = _blank("[missing_prediction]")
            continue
        key = {"judge_fp": fingerprint, "model": ctx.model_name,
               "dataset": ctx.dataset_key, "index": index}
        cached = cache.get(key) if cache else None
        if cached is not None:
            results[index] = cached
        else:
            pending.append((index, row))

    with ThreadPoolExecutor(max_workers=max(1, ctx.max_workers)) as executor:
        futures = {
            executor.submit(_judge_one, client, prompt, row, image_map.get(index) or row.get("image")): index
            for index, row in pending
        }
        for future in as_completed(futures):
            index = futures[future]
            result = future.result()
            results[index] = result
            # 判失败绝不进缓存：缓存是 append-only，一次超时会被永久冻成一个分数，
            # 重跑也修不好。
            if cache and not str(result.get("judge_reason", "")).startswith("[judge_error]"):
                cache.set({"judge_fp": fingerprint, "model": ctx.model_name,
                           "dataset": ctx.dataset_key, "index": index}, result)

    out = scored.copy()
    ordered = [results.get(str(row.get("index", i)), _blank("[missing_result]"))
               for i, row in enumerate(rows)]
    out["hit"] = [r.get("hit") for r in ordered]
    for dim in DIMENSIONS:
        out[f"judge_{dim}"] = [r.get(f"judge_{dim}") for r in ordered]
    out["judge_reason"] = [str(r.get("judge_reason", "")) for r in ordered]
    return out


def _judge_one(client: Any, prompt: str, row: Mapping[str, Any], image: Any) -> dict[str, Any]:
    user_text = (
        f"描述任务：{row.get('question', '')}\n"
        f"参考描述：{row.get('answer', '')}\n"
        f"被测模型回答：{row.get('prediction', '')}\n\n"
        "请看着图片按三个维度打分，只输出 JSON。"
    )
    last_error = ""
    for _ in range(2):
        try:
            content = client.judge_raw(prompt, user_text, image_b64=image)
            return parse_describe(content).as_columns()
        except Exception as exc:  # noqa: BLE001 - 判词千奇百怪，原样带回报表
            last_error = f"{type(exc).__name__}: {exc}"
    # 判不出来记 hit=None 而不是 0：判失败和判低分是两件事，记 0 会在两个模型
    # 失败率不同时把对比拉出方向性偏差。
    return _blank(f"[judge_error] {last_error}")


def _blank(reason: str) -> dict[str, Any]:
    blank: dict[str, Any] = {"hit": None, "judge_reason": reason}
    blank.update({f"judge_{dim}": None for dim in DIMENSIONS})
    return blank


def _prompt_text(ctx: ScoringContext) -> str:
    configured = ctx.params.get("judge_prompt")
    path = resolve_path(ctx, configured) if configured else None
    if path is None:
        from pathlib import Path

        path = Path(__file__).resolve().parents[2] / DEFAULT_PROMPT
    return load_prompt_text(path)
