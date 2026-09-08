"""自由文本答案：交给裁判模型 pointwise 打分。engine=judge，结果带裁判噪声。"""

from __future__ import annotations

import pandas as pd

from ..metrics_text import aux_metrics
from ..score_vqa import score_pointwise_vqa
from . import JUDGE, ScoringContext, register


@register("judge_text", engine=JUDGE, pairwise=True)
def score_judge_text(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    if ctx.do_pointwise:
        return score_pointwise_vqa(
            data,
            ctx.model_name,
            ctx.judge_client,
            cache=ctx.judge_cache,
            workers=ctx.max_workers,
            image_map=dict(ctx.image_map),
            dataset_key=ctx.dataset_key,
        )
    # 关了 pointwise 时不打分，只算几个不花钱的文本重合度指标，hit 留空。
    # hit=NA 而不是 0：没判过和判错是两回事，填 0 会被汇总当成「全错」。
    scored = data.copy()
    scored["hit"] = pd.NA
    aux = [aux_metrics(row.get("answer", ""), row.get("prediction", "")) for _, row in scored.iterrows()]
    for col in ("bleu1", "bleu2", "rouge_l", "pred_len"):
        scored[col] = [m[col] for m in aux]
    return scored
