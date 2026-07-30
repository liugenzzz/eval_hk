from __future__ import annotations

import math
from collections import Counter


def tokenize_for_overlap(text: object) -> list[str]:
    text = str(text or "").lower()
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                tokens.append("".join(buf))
                buf = []
            tokens.append(ch)
        elif ch.isascii() and ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf = []
    if buf:
        tokens.append("".join(buf))
    return tokens


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)]


def bleu_score(reference: object, prediction: object, max_n: int = 1) -> float:
    ref = tokenize_for_overlap(reference)
    pred = tokenize_for_overlap(prediction)
    if not pred or not ref:
        return 0.0
    bp = 1.0 if len(pred) > len(ref) else math.exp(1.0 - len(ref) / max(len(pred), 1))
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(_ngrams(pred, n))
        ref_ngrams = Counter(_ngrams(ref, n))
        total = sum(pred_ngrams.values())
        if total == 0:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, ref_ngrams.get(gram, 0)) for gram, count in pred_ngrams.items())
        if n == 1:
            precisions.append(overlap / total)
        else:
            precisions.append((overlap + 1.0) / (total + 1.0))
    if any(p <= 0 for p in precisions):
        return 0.0
    return float(bp * math.exp(sum(math.log(p) for p in precisions) / max_n))


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l_score(reference: object, prediction: object) -> float:
    ref = tokenize_for_overlap(reference)
    pred = tokenize_for_overlap(prediction)
    if not ref or not pred:
        return 0.0
    lcs = _lcs_len(ref, pred)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def aux_metrics(reference: object, prediction: object) -> dict[str, float | int]:
    pred_tokens = tokenize_for_overlap(prediction)
    return {
        "bleu1": round(bleu_score(reference, prediction, max_n=1), 4),
        "bleu2": round(bleu_score(reference, prediction, max_n=2), 4),
        "rouge_l": round(rouge_l_score(reference, prediction), 4),
        "pred_len": int(len(pred_tokens)),
    }
