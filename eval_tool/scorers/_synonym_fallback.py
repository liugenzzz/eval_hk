"""C / E 组共用的同义兜底调度：只对代码判不了的那些行问裁判。

**默认不启用。** 要打开就在 params 里写 ``"judge_synonym": true``（提示词可以用
``judge_synonym_prompt`` 换一份）。关着的时候这两组是纯代码打分器，重跑一百遍逐位
相同；打开之后它们的结果会带一点裁判噪声，报表里 ``judge_*`` 那几列说明哪些行走过
裁判、走出了什么结论。

**触发的行本来就要记 0**（代码判不了 = 判错）。裁判只可能把 0 抬成 1，不会反过来，
所以关着跑出来的数是一个**偏严的下界**。报告里得写清楚这次跑的是哪一种。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import pandas as pd

from ..prompting import load_prompt_text
from ..synonym import HYPERNYM, HYPONYM, ask
from . import ScoringContext, resolve_path

DEFAULT_PROMPT = "prompts/judge_synonym_v1.txt"
COLUMNS = ("judge_same", "judge_relation", "judge_reason")


def enabled(ctx: ScoringContext) -> bool:
    return bool(ctx.params.get("judge_synonym")) and ctx.judge_client is not None


def prompt_text(ctx: ScoringContext) -> str:
    configured = ctx.params.get("judge_synonym_prompt")
    if configured:
        return load_prompt_text(resolve_path(ctx, configured))
    from pathlib import Path

    return load_prompt_text(Path(__file__).resolve().parents[2] / DEFAULT_PROMPT)


def apply(
    scored: pd.DataFrame,
    ctx: ScoringContext,
    gold_of: Callable[[Any], str],
    pred_of: Callable[[Any], str],
    *,
    promote_relations: tuple[str, ...] = (),
) -> pd.DataFrame:
    """对 ``judge_fallback_needed == 1`` 的行问裁判，把判为同义的抬成命中。

    ``promote_relations`` 决定哪些关系算「命中主指标」。E 组只认 ``exact`` ——
    裁判说是上位词或下位词，那和代码判出来的上位/下位命中是同一件事，各自计数，
    **不并进精确命中率**，否则「答粗一点更安全」这种退化就被裁判洗白了。
    """
    out = scored.copy()
    for column in COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA
    if "judge_fallback_needed" not in out.columns:
        return out
    pending = out.index[out["judge_fallback_needed"] == 1].tolist()
    if not enabled(ctx) or not pending:
        return out

    prompt = prompt_text(ctx)
    cache = ctx.judge_cache
    fingerprint = getattr(getattr(ctx.judge_client, "settings", None), "fingerprint", "")
    results: dict[Any, dict[str, Any]] = {}
    to_ask: list[Any] = []
    for position in pending:
        key = {"judge_fp": fingerprint, "model": ctx.model_name,
               "dataset": f"{ctx.dataset_key}#synonym", "index": str(out.at[position, "index"])
               if "index" in out.columns else str(position)}
        cached = cache.get(key) if cache else None
        if cached is not None:
            results[position] = cached
        else:
            to_ask.append((position, key))

    with ThreadPoolExecutor(max_workers=max(1, ctx.max_workers)) as executor:
        futures = {
            executor.submit(ask, ctx.judge_client, prompt,
                            gold_of(out.loc[position]), pred_of(out.loc[position])): (position, key)
            for position, key in to_ask
        }
        for future in as_completed(futures):
            position, key = futures[future]
            verdict = future.result()
            results[position] = verdict
            if cache and not str(verdict.get("judge_reason", "")).startswith("[judge_error]"):
                cache.set(key, verdict)

    for position, verdict in results.items():
        for column in COLUMNS:
            out.at[position, column] = verdict.get(column)
        relation = str(verdict.get("judge_relation") or "")
        if verdict.get("judge_same") and relation in promote_relations:
            out.at[position, "hit"] = 1
        # 裁判说是上位/下位词的，并进代码那两列各自计数，不进主指标。
        if verdict.get("judge_same") and relation == HYPERNYM and "hit_hypernym" in out.columns:
            out.at[position, "hit_hypernym"] = 1
        if verdict.get("judge_same") and relation == HYPONYM and "hit_hyponym" in out.columns:
            out.at[position, "hit_hyponym"] = 1
    return out
