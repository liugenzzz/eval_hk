from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .cache import JsonlCache
from .history import format_history_for_judge, parse_history_cell
from .judge import JudgeClient
from .metrics_text import aux_metrics


@dataclass(frozen=True)
class PairwiseMerge:
    winner: str
    outcome_for_a: str


def merge_pairwise_decisions(model_a: str, model_b: str, forward_winner: str, reverse_winner: str) -> PairwiseMerge:
    forward_actual = _actual_winner(forward_winner, slot_a=model_a, slot_b=model_b)
    reverse_actual = _actual_winner(reverse_winner, slot_a=model_b, slot_b=model_a)
    if forward_actual == reverse_actual and forward_actual != "tie":
        winner = forward_actual
    else:
        winner = "tie"
    if winner == model_a:
        outcome = "win"
    elif winner == model_b:
        outcome = "loss"
    else:
        outcome = "tie"
    return PairwiseMerge(winner=winner, outcome_for_a=outcome)


def _actual_winner(winner: str, slot_a: str, slot_b: str) -> str:
    normalized = str(winner or "tie").strip().upper()
    if normalized == "A":
        return slot_a
    if normalized == "B":
        return slot_b
    return "tie"


def _tqdm(iterable: Any, **kwargs: Any) -> Any:
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return iterable


def _judge_pointwise_with_retry(
    client: JudgeClient,
    row: dict[str, Any],
    image_b64: str | None,
) -> dict[str, Any]:
    question = format_history_for_judge(parse_history_cell(row.get("history")), row.get("question", ""))
    last_err = ""
    for _ in range(client.settings.max_retries):
        try:
            result = client.judge_pointwise(question, row.get("answer", ""), row.get("prediction", ""), image_b64)
            return {
                "hit": result["hit"],
                "judge_reason": result["reason"],
                "quality_score": result.get("quality_score"),
                "param_score": result.get("param_score"),
                "fact_score": result.get("fact_score"),
                "visual_score": result.get("visual_score"),
                "fabrication_score": result.get("fabrication_score"),
                "style_score": result.get("style_score"),
            }
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return {
        "hit": 0,
        "judge_reason": f"[judge_error] {last_err}",
        "quality_score": None,
        "param_score": None,
        "fact_score": None,
        "visual_score": None,
        "fabrication_score": None,
        "style_score": None,
    }


def score_pointwise_vqa(
    data: pd.DataFrame,
    model_name: str,
    client: JudgeClient,
    cache: JsonlCache | None = None,
    workers: int = 8,
    image_map: dict[str, str] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    image_map = image_map or {}
    rows = data.to_dict("records")
    scored = data.copy()
    results: dict[str, dict[str, Any]] = {}
    pending: list[tuple[str, dict[str, Any]]] = []
    for i, row in enumerate(rows):
        idx = str(row.get("index", i))
        prediction = row.get("prediction", "")
        if pd.isna(prediction) or str(prediction).strip() == "":
            results[idx] = {"hit": 0, "judge_reason": "[missing_prediction]"}
            continue
        key = {"model": model_name, "dataset": "vqa", "index": idx}
        cached = cache.get(key) if cache else None
        if cached is not None:
            results[idx] = cached
        else:
            pending.append((idx, row))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {}
        for idx, row in pending:
            image_b64 = image_map.get(idx) or row.get("image")
            fut = executor.submit(_judge_pointwise_with_retry, client, row, image_b64)
            futures[fut] = (idx, row)
        iterator = as_completed(futures)
        if progress and futures:
            iterator = _tqdm(iterator, total=len(futures), desc=f"VQA pointwise {model_name}")
        for fut in iterator:
            idx, _ = futures[fut]
            result = fut.result()
            results[idx] = result
            if cache:
                cache.set({"model": model_name, "dataset": "vqa", "index": idx}, result)

    hits = []
    reasons = []
    quality_scores = []
    param_scores = []
    fact_scores = []
    visual_scores = []
    fabrication_scores = []
    style_scores = []
    aux = []
    for i, row in enumerate(rows):
        idx = str(row.get("index", i))
        result = results.get(idx, {"hit": 0, "judge_reason": "[missing_result]"})
        # hit may be a 0/1 legacy/thresholded verdict or a 0-1 weighted quality score
        # (see judge.parse_pointwise_judge) -- must stay float, int() would truncate
        # any fractional weighted score down to 0.
        hits.append(float(result.get("hit", 0)))
        reasons.append(str(result.get("judge_reason", "")))
        quality_scores.append(result.get("quality_score"))
        param_scores.append(result.get("param_score"))
        fact_scores.append(result.get("fact_score"))
        visual_scores.append(result.get("visual_score"))
        fabrication_scores.append(result.get("fabrication_score"))
        style_scores.append(result.get("style_score"))
        aux.append(aux_metrics(row.get("answer", ""), row.get("prediction", "")))
    scored["hit"] = hits
    scored["judge_reason"] = reasons
    scored["quality_score"] = quality_scores
    scored["param_score"] = param_scores
    scored["visual_score"] = visual_scores
    scored["fabrication_score"] = fabrication_scores
    scored["fact_score"] = fact_scores
    scored["style_score"] = style_scores
    for col in ("bleu1", "bleu2", "rouge_l", "pred_len"):
        scored[col] = [m[col] for m in aux]
    return scored


def _judge_pairwise_with_retry(
    client: JudgeClient,
    question: object,
    reference: object,
    answer_a: object,
    answer_b: object,
    image_b64: str | None,
) -> dict[str, Any]:
    last_err = ""
    for _ in range(client.settings.max_retries):
        try:
            winner, reason = client.judge_pairwise(question, reference, answer_a, answer_b, image_b64)
            return {"winner": winner, "reason": reason}
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return {"winner": "tie", "reason": f"[judge_error] {last_err}"}


def score_pairwise_vs_baseline(
    model_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    model_name: str,
    baseline_model: str,
    client: JudgeClient,
    cache: JsonlCache | None = None,
    workers: int = 8,
    image_map: dict[str, str] | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    image_map = image_map or {}
    left = model_df.copy()
    right = baseline_df[["index", "prediction", "pred_len"]].copy()
    right = right.rename(columns={"prediction": "baseline_prediction", "pred_len": "baseline_pred_len"})
    merged = left.merge(right, on="index", how="inner")
    jobs: list[tuple[str, str, dict[str, Any]]] = []
    raw: dict[tuple[str, str], dict[str, Any]] = {}

    for _, row in merged.iterrows():
        idx = str(row["index"])
        for direction in ("forward", "reverse"):
            key = {"model_A": model_name, "model_B": baseline_model, "index": idx, "direction": direction}
            cached = cache.get(key) if cache else None
            if cached is not None:
                raw[(idx, direction)] = cached
            else:
                jobs.append((idx, direction, row.to_dict()))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {}
        for idx, direction, row in jobs:
            image_b64 = image_map.get(str(idx)) or row.get("image")
            if direction == "forward":
                answer_a = row.get("prediction", "")
                answer_b = row.get("baseline_prediction", "")
            else:
                answer_a = row.get("baseline_prediction", "")
                answer_b = row.get("prediction", "")
            fut = executor.submit(
                _judge_pairwise_with_retry,
                client,
                format_history_for_judge(parse_history_cell(row.get("history")), row.get("question", "")),
                row.get("answer", ""),
                answer_a,
                answer_b,
                image_b64,
            )
            futures[fut] = (idx, direction)
        iterator = as_completed(futures)
        if progress and futures:
            iterator = _tqdm(iterator, total=len(futures), desc=f"Pairwise {model_name} vs {baseline_model}")
        for fut in iterator:
            idx, direction = futures[fut]
            result = fut.result()
            raw[(str(idx), direction)] = result
            if cache:
                cache.set({"model_A": model_name, "model_B": baseline_model, "index": str(idx), "direction": direction}, result)

    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        idx = str(row["index"])
        forward = raw.get((idx, "forward"), {"winner": "tie", "reason": "[missing_forward]"})
        reverse = raw.get((idx, "reverse"), {"winner": "tie", "reason": "[missing_reverse]"})
        merged_decision = merge_pairwise_decisions(model_name, baseline_model, forward.get("winner", "tie"), reverse.get("winner", "tie"))
        pred_len = int(row.get("pred_len", 0) or 0)
        baseline_len = int(row.get("baseline_pred_len", 0) or 0)
        rows.append(
            {
                "index": idx,
                "model": model_name,
                "baseline_model": baseline_model,
                "winner": merged_decision.winner,
                "outcome_for_model": merged_decision.outcome_for_a,
                "forward_winner": forward.get("winner", "tie"),
                "reverse_winner": reverse.get("winner", "tie"),
                "forward_reason": forward.get("reason", ""),
                "reverse_reason": reverse.get("reason", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "prediction": row.get("prediction", ""),
                "baseline_prediction": row.get("baseline_prediction", ""),
                "category": row.get("category", ""),
                "l2-category": row.get("l2-category", ""),
                "source_id": row.get("source_id", ""),
                "pred_len": pred_len,
                "baseline_pred_len": baseline_len,
                "len_diff": pred_len - baseline_len,
            }
        )
    return pd.DataFrame(rows)
