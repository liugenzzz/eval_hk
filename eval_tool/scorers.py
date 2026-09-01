"""打分器注册表：数据集声明 kind，按 kind 查表分派。

早先 run_eval 里是这么写的：

    for dataset_key in ("mcq", "judge", "vqa"):     # 硬编码三元组
        ...
        if dataset_key in {"mcq", "judge"}:         # 硬编码分支
            scored = score_choice_dataframe(...)
        else:
            scored = score_pointwise_vqa(...)

datasets 的键名本来是自由的（config.py 的 DEFAULT_DATASETS 只是默认值），但打分
逻辑写死了这三个 —— 自定义键名连循环都进不去，静默被忽略。再往里加检测框、
多框、短答案、物体识别几种打分方式，run_eval 会变成六七个分支的巨型函数。

现在：数据集声明自己的 kind，打分器按 kind 注册、查表分派。加一种打分方式
只要写一个函数加一行 @register，run_eval 一个字都不用改。

【向后兼容】未声明 kind 时按键名推断，规则和原来的分支完全一致：
mcq / judge -> choice，其余 -> judge_text。所以老配置行为不变。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pandas as pd

CHOICE = "choice"
JUDGE_TEXT = "judge_text"

# 未声明 kind 时按键名推断。这两个键名走选项抽取，其余一律走裁判 ——
# 和重构前 `if dataset_key in {"mcq", "judge"}` 那个分支一字不差。
_KIND_BY_KEY = {"mcq": CHOICE, "judge": CHOICE}


def default_kind(dataset_key: str) -> str:
    return _KIND_BY_KEY.get(dataset_key, JUDGE_TEXT)


@dataclass
class ScoreContext:
    """打分器可能用到的全部上下文。

    不同 kind 需要的东西差很多：choice 只要知道是哪个数据集，judge_text 要裁判
    客户端、缓存、并发数和图片表，几何类打分器（坐标偏差、IoU）一个都不需要。
    与其给每个打分器设计不同签名，不如统一传这一个对象，各取所需。
    """

    dataset_key: str
    model_name: str
    config: Any = None
    judge_client: Any = None
    pointwise_cache: Any = None
    image_map: Optional[Dict[str, Any]] = None


Scorer = Callable[[pd.DataFrame, ScoreContext], pd.DataFrame]

_REGISTRY: Dict[str, Scorer] = {}


def register(kind: str) -> Callable[[Scorer], Scorer]:
    """注册一个打分器。重名直接报错 —— 静默覆盖会让人对着一份没在跑的代码调。"""

    def wrap(fn: Scorer) -> Scorer:
        if kind in _REGISTRY:
            raise ValueError(f"打分器 kind 重复注册：{kind}")
        _REGISTRY[kind] = fn
        return fn

    return wrap


def get(kind: str) -> Scorer:
    if kind not in _REGISTRY:
        raise KeyError(
            f"没有 kind 为 {kind!r} 的打分器。已注册：{sorted(_REGISTRY)}。"
            f"数据集的 kind 在配置的 dataset_kinds 里声明。"
        )
    return _REGISTRY[kind]


def registered_kinds() -> list[str]:
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------
# 内置打分器。行为与重构前完全一致，只是从 if/else 挪进了注册表。
# --------------------------------------------------------------------------
@register(CHOICE)
def _score_choice(data: pd.DataFrame, ctx: ScoreContext) -> pd.DataFrame:
    """选项抽取（mcq 的 A/B/C/D、judge 的是否）。纯代码，不调模型。"""
    from .score_mcq import score_choice_dataframe

    return score_choice_dataframe(data, dataset=ctx.dataset_key)


@register(JUDGE_TEXT)
def _score_judge_text(data: pd.DataFrame, ctx: ScoreContext) -> pd.DataFrame:
    """开放问答，交给 LLM 裁判逐条打分。

    config.do_pointwise 关闭时不调裁判，只算 BLEU/ROUGE 这些辅助指标 ——
    这条路径是为了「先看看预测长什么样」，不花裁判调用。
    """
    from .metrics_text import aux_metrics
    from .score_vqa import score_pointwise_vqa

    if ctx.config is not None and not ctx.config.do_pointwise:
        scored = data.copy()
        scored["hit"] = pd.NA
        aux = [
            aux_metrics(row.get("answer", ""), row.get("prediction", ""))
            for _, row in scored.iterrows()
        ]
        for col in ("bleu1", "bleu2", "rouge_l", "pred_len"):
            scored[col] = [m[col] for m in aux]
        return scored

    return score_pointwise_vqa(
        data,
        ctx.model_name,
        ctx.judge_client,
        cache=ctx.pointwise_cache,
        workers=ctx.config.max_workers if ctx.config is not None else 1,
        image_map=ctx.image_map,
    )
