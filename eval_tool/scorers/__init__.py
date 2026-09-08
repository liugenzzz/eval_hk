"""打分器注册表：数据集的 kind -> 打分实现。

为什么要有这一层：run_eval 以前把数据集键写死成 ``("mcq", "judge", "vqa")`` 的
循环，再用 ``if dataset_key in {"mcq", "judge"}`` 挑打分分支。每新增一种答案形态
（画框、多框、计数、物体识别……）就要再加一个 elif，最后是一个六分支的巨型函数；
更糟的是数据集的**键名**和**打分方式**被焊死在一起 —— 换一套数据就得改代码。

现在数据集配置里声明 ``kind``，打分器按 kind 查表分派::

    "datasets": {
      "vqa":    {"name": "aero_vqa", "kind": "judge_text"},
      "ground": {"name": "eval_set_v1", "kind": "grounding_single",
                 "params": {"iou_gate": 0.5}}
    }

kind 只回答「这种答案形态怎么打分」，不回答「这批数据是什么」。同一个 kind 可以
被任意多个数据集复用，同一批数据换个 kind 就换了口径，两边互不牵连。

``engine`` 把「代码打分」和「裁判打分」分开标注：``engine="code"`` 的打分器**不许
调任何模型**，同一份预测重跑一百遍必须逐位相同。报表按这个字段区分哪些数字是
可复现的硬指标、哪些带裁判噪声。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import pandas as pd

CODE = "code"
JUDGE = "judge"


@dataclass(frozen=True)
class ScoringContext:
    """打分器拿得到的全部上下文。

    刻意不传整个 EvalConfig：打分器只该看见它自己需要的东西，看得见配置就迟早
    会有人从里面掏一个别的数据集的字段出来用。
    """

    dataset_key: str
    kind: str
    model_name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    # 只有 engine="judge" 的打分器会用到下面四个
    judge_client: Any | None = None
    judge_cache: Any | None = None
    image_map: Mapping[str, str] = field(default_factory=dict)
    do_pointwise: bool = True
    max_workers: int = 8


Scorer = Callable[[pd.DataFrame, ScoringContext], pd.DataFrame]


@dataclass(frozen=True)
class ScorerSpec:
    kind: str
    score: Scorer
    engine: str = CODE
    # 该 kind 的答案是否值得做 base vs sft 的成对裁判。有确定答案的形态
    # （选择题、坐标、类别名）pointwise 的绝对分就够了，pairwise 纯属浪费裁判调用。
    pairwise: bool = False

    @property
    def needs_judge(self) -> bool:
        return self.engine == JUDGE


_REGISTRY: dict[str, ScorerSpec] = {}


class UnknownKindError(KeyError):
    """数据集声明了一个没有实现的 kind。"""


def register(kind: str, *, engine: str = CODE, pairwise: bool = False) -> Callable[[Scorer], Scorer]:
    """把一个打分函数登记到 kind 上。重复注册直接报错，不静默覆盖。"""

    def decorator(fn: Scorer) -> Scorer:
        if kind in _REGISTRY:
            raise ValueError(f"打分器 kind 重复注册：{kind}")
        if engine not in (CODE, JUDGE):
            raise ValueError(f"engine 只能是 {CODE!r} 或 {JUDGE!r}，得到 {engine!r}")
        _REGISTRY[kind] = ScorerSpec(kind=kind, score=fn, engine=engine, pairwise=pairwise)
        return fn

    return decorator


def get(kind: str) -> ScorerSpec:
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise UnknownKindError(
            f"未知的数据集 kind：{kind!r}。已实现：{', '.join(known_kinds()) or '（无）'}"
        ) from None


def is_registered(kind: str) -> bool:
    return kind in _REGISTRY


def known_kinds() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# 注册副作用靠 import 触发，放在文件末尾避免循环导入。
from . import choice as _choice  # noqa: E402,F401
from . import grounding as _grounding  # noqa: E402,F401
from . import judge_text as _judge_text  # noqa: E402,F401
