"""选择题 / 判断题：从模型输出里抽出选项字母再比对。纯代码，不调模型。"""

from __future__ import annotations

import pandas as pd

from ..score_mcq import score_choice_dataframe
from ..compliance import CHOICE
from . import CODE, ScoringContext, register


@register("choice", engine=CODE, answer_form=CHOICE)
def score_choice(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    # 判断题的合法选项是 A/B，选择题是 A/B/C/D。历史上这件事是按数据集键名
    # ("judge") 判的，所以键名默认继续生效；新数据集应当显式写
    # params.choice_style，别再靠键名取巧。
    style = str(ctx.params.get("choice_style") or ctx.dataset_key)
    return score_choice_dataframe(data, dataset=style)
