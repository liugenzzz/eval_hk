#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
开放题(VQA)描述性任务自定义数据集 —— 复用 VLMEvalKit 官方 evaluate 钩子 + LLM 裁判打分
============================================================
解决的问题：选择/判断题 VLMEvalKit 能精确匹配自动判分；开放题答案是描述，精确匹配无效，
必须挂一个 LLM 裁判按语义打分。本文件给 aero_vqa 数据集补上这步——裁判用本地 qwen3.6-27b，
评分二值（描述忠实、具体、覆盖关键可见要点=对），并按 category(能力) / l2-category(内容类型)
聚合，外加能力×内容类型交叉表。另输出 BLEU-1 / BLEU-2 / ROUGE-L / pred_len 作为辅助分析指标，
用于观察词面覆盖和回答长度，主指标仍是 LLM 裁判 hit。

【怎么用】
1. 把本文件放进你的 VLMEvalKit 工程（和其它数据集类一起，或单独一个文件）。
2. 在运行入口 import 进来（让 VLMEvalKit 认识 AeroVQADataset 这个数据集类）：
       from aero_vqa_dataset import AeroVQADataset
3. 把 aero_vqa.tsv 放进 $LMUData。
4. 跑：python run.py --data aero_vqa --model qwen3_vl_8b_instruct --work-dir results_aero

【裁判模型】本文件里的 evaluate 自带裁判调用（直连你的 qwen3.6-27b 端点），不依赖 VLMEvalKit 的
judge 插件配置，更稳、更可控。裁判端点在下方 CONFIG 改。

⚠ 版本相关（务必核对你本机 vlmeval 版本）：
  - 顶部 from vlmeval... 的导入路径、ImageBaseDataset.build_prompt 的签名、evaluate 的返回/落盘约定，
    不同版本略有差异。本文件按主流版本写，跑不通时对照你版本的 vlmeval/dataset/image_vqa.py
    （官方长答案 LLM-judge 数据集，如 MMVet）改这几处。
  - evaluate 必须返回一个指标 DataFrame，并把分数落盘；本文件已按此实现。
"""

import json
import math
import os
import re
import threading
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== CONFIG（裁判，改这里）====================
JUDGE_API_BASE = "http://192.168.48.7:18180/v1/chat/completions"
JUDGE_API_KEY  = "sk-local"
JUDGE_MODEL    = "qwen3.6-27b"
JUDGE_WORKERS  = 8       # 裁判并发
JUDGE_RETRIES  = 3
JUDGE_TIMEOUT  = 120
JUDGE_TEMP     = 0.0
# 评分尺度：当前实现是二值（对/错→1/0），聚合即准确率。要改 0~N 分见 _parse_judge。
# ============================================================

# 裁判评分提示词（看图评分，领域化，二值，中性长度；要改标准就改这里）
# 关键：裁判会看到图片，以图为准判断"准确+切题具体"，而非"是否复刻参考描述"，
# 从根上消除"啰嗦占便宜/简洁被冤枉"的长度偏差。
VQA_JUDGE_PROMPT = """你是航空工程领域的图像描述评分专家。我会给你一张航空技术图片、一道关于它的描述任务、一份参考描述、以及被测模型的回答。请你**看着图片**判断被测模型的回答是否合格。

重要：参考描述只是一种合理答案，可能不完整、本身也可能较粗。判分以**图片**为准，不是以"是否复刻参考描述"为准。

判为合格 correct=true 需**同时**满足：
1. 准确：回答中关于图里对象、结构/位置关系、标注、数值、部件名称的描述，与图片一致；没有编造图中不存在的内容，也没有明显错误。
2. 切题且具体：回答确实针对这张图的可见内容作答，点到了与问题相关的关键可见信息，而不是只讲通用功能或航空常识。

判为不合格 correct=false 的情形：
- 编造/臆测图中没有的部件、数值、标注或关系（哪怕说得很多很详细）。
- 只泛泛而谈通用功能（如"用于连接/支撑/传递载荷"），没有落到这张图的具体可见内容。
- 答非所问、只复述问题、或模糊到无法判断是否真看懂了图。

长度中性（重要）：
- 不要因为回答长、信息多就更容易判对；冗长但夹带错误或编造的，判 false。
- 不要因为回答短就判 false；只要准确且点到了关键可见信息，简洁的回答也判 true。

只输出一个 JSON，无多余文字、无 markdown：
{"correct": true 或 false, "reason": "一句话理由，指出依据图片的关键点"}"""


def _judge_call(question, reference, prediction, image_b64=None, mime="image/jpeg"):
    """调裁判模型给一条开放题打分（看图）。返回 (hit:int 0/1, reason:str)。失败抛异常（外层重试）。"""
    text = (f"描述任务：{question}\n"
            f"参考描述：{reference}\n"
            f"被测模型回答：{prediction}\n\n请看着图片，按描述性任务标准判断是否合格，只输出 JSON。")
    if image_b64:
        user_content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        ]
    else:
        # 拿不到图时退回纯文本判分（不理想，仅兜底）
        user_content = text
    body = json.dumps({
        "model": JUDGE_MODEL,
        "messages": [{"role": "system", "content": VQA_JUDGE_PROMPT},
                     {"role": "user", "content": user_content}],
        "temperature": JUDGE_TEMP, "max_tokens": 256,
    }).encode("utf-8")
    req = urllib.request.Request(JUDGE_API_BASE, data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {JUDGE_API_KEY}"})
    with urllib.request.urlopen(req, timeout=JUDGE_TIMEOUT) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return _parse_judge(out["choices"][0]["message"]["content"])


def _parse_judge(content):
    """解析裁判输出 → (hit 0/1, reason)。容忍 markdown/多余文字。"""
    content = content.strip()
    content = re.sub(r"^```(?:json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        content = m.group(0)
    obj = json.loads(content)
    hit = 1 if obj.get("correct") is True else 0
    return hit, obj.get("reason", "")


def _tokenize_for_overlap(text):
    """轻量中英混合切分：中文按字，连续英文/数字按词；避免依赖 jieba/nltk。"""
    text = str(text or "").lower()
    tokens = []
    buf = []
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


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(0, len(tokens) - n + 1)]


def _bleu_score(reference, prediction, max_n=1):
    """句级 BLEU-N，带简单平滑；仅作辅助词面覆盖指标。"""
    ref = _tokenize_for_overlap(reference)
    pred = _tokenize_for_overlap(prediction)
    if not pred:
        return 0.0
    if not ref:
        return 0.0
    bp = 1.0 if len(pred) > len(ref) else math.exp(1.0 - len(ref) / max(len(pred), 1))
    precisions = []
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


def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _rouge_l_score(reference, prediction):
    """ROUGE-L F1，辅助衡量描述与参考的序列覆盖。"""
    ref = _tokenize_for_overlap(reference)
    pred = _tokenize_for_overlap(prediction)
    if not ref or not pred:
        return 0.0
    lcs = _lcs_len(ref, pred)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _aux_metrics(reference, prediction):
    pred_tokens = _tokenize_for_overlap(prediction)
    return {
        "bleu1": round(_bleu_score(reference, prediction, max_n=1), 4),
        "bleu2": round(_bleu_score(reference, prediction, max_n=2), 4),
        "rouge_l": round(_rouge_l_score(reference, prediction), 4),
        "pred_len": int(len(pred_tokens)),
    }


def _judge_one_row(idx, question, reference, prediction, judge_fn, image_b64=None):
    last_err = ""
    for _ in range(JUDGE_RETRIES):
        try:
            hit, reason = judge_fn(question, reference, prediction, image_b64)
            return {"index": idx, "hit": hit, "judge_reason": reason}
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    # 裁判一直失败：记 0 并标注，不静默
    return {"index": idx, "hit": 0, "judge_reason": f"[judge_error] {last_err}"}


def score_dataframe(data, judge_fn=_judge_call, workers=JUDGE_WORKERS, progress=True, img_map=None):
    """
    核心：给开放题结果打分并聚合（看图判分）。可独立测试（不依赖 vlmeval）。
    入参 data: 含列 question / answer / prediction / category / l2-category（pandas DataFrame）。
    img_map: {index: 图片base64}，裁判看图用；为空时尝试用 data 里的 image 列，再不行退纯文本。
    返回 (scored_df, metrics_dict)。
      scored_df: 原表 + hit + judge_reason + bleu1/bleu2/rouge_l/pred_len。
      metrics_dict: overall / BLEU / ROUGE / by_capability / by_content_type / cross(能力×内容类型)。
    """
    import pandas as pd
    img_map = img_map or {}

    # 统一列名：l2-category 兼容 l2_category
    l2col = "l2-category" if "l2-category" in data.columns else (
            "l2_category" if "l2_category" in data.columns else None)
    catcol = "category" if "category" in data.columns else None

    rows = data.to_dict("records")
    results = {}
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **k): return x

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for i, r in enumerate(rows):
            idx = r.get("index", i)
            img_b64 = img_map.get(idx) or r.get("image")   # 优先 img_map，其次行内 image 列
            futs.append(ex.submit(_judge_one_row, idx,
                                  str(r.get("question", "")), str(r.get("answer", "")),
                                  str(r.get("prediction", "")), judge_fn, img_b64))
        it = as_completed(futs)
        if progress:
            it = tqdm(it, total=len(futs), desc="VQA 裁判打分")
        for fut in it:
            res = fut.result()
            with lock:
                results[res["index"]] = res

    data = data.copy()
    data["hit"] = [results.get(r.get("index", i), {}).get("hit", 0) for i, r in enumerate(rows)]
    data["judge_reason"] = [results.get(r.get("index", i), {}).get("judge_reason", "") for i, r in enumerate(rows)]
    aux = [_aux_metrics(str(r.get("answer", "")), str(r.get("prediction", ""))) for r in rows]
    data["bleu1"] = [m["bleu1"] for m in aux]
    data["bleu2"] = [m["bleu2"] for m in aux]
    data["rouge_l"] = [m["rouge_l"] for m in aux]
    data["pred_len"] = [m["pred_len"] for m in aux]

    metrics = {
        "overall": round(float(data["hit"].mean()), 4),
        "n": int(len(data)),
        "bleu1": round(float(data["bleu1"].mean()), 4),
        "bleu2": round(float(data["bleu2"].mean()), 4),
        "rouge_l": round(float(data["rouge_l"].mean()), 4),
        "avg_pred_len": round(float(data["pred_len"].mean()), 2),
    }
    if catcol:
        metrics["by_capability"] = {k: round(float(v), 4)
                                    for k, v in data.groupby(catcol)["hit"].mean().items()}
    if l2col:
        metrics["by_content_type"] = {k: round(float(v), 4)
                                      for k, v in data.groupby(l2col)["hit"].mean().items()}
    if catcol and l2col:
        cross = data.pivot_table(index=catcol, columns=l2col, values="hit", aggfunc="mean")
        metrics["cross"] = cross.round(4).to_dict(orient="index")  # {能力: {内容类型: 正确率}}
    return data, metrics


# ============================================================
# VLMEvalKit 自定义数据集类（vlmeval 的导入做成惰性，便于本文件单测打分逻辑）
# ============================================================
def _get_base():
    from vlmeval.dataset.image_base import ImageBaseDataset
    return ImageBaseDataset


# 用工厂函数延迟继承，避免 import 阶段就要求装好 vlmeval
def _build_class():
    ImageBaseDataset = _get_base()
    from vlmeval.smp import load, dump

    class AeroVQADataset(ImageBaseDataset):
        TYPE = "VQA"
        # 本地数据集，无需下载；占位即可
        DATASET_URL = {"aero_vqa": ""}
        DATASET_MD5 = {"aero_vqa": ""}

        def load_data(self, dataset):
            lmu = os.path.expanduser(os.environ.get("LMUData", "~/LMUData"))
            return load(os.path.join(lmu, f"{dataset}.tsv"))

        def build_prompt(self, line):
            # 复用基类的"图片 + 问题"组装；可按需追加作答指令
            msgs = ImageBaseDataset.build_prompt(self, line)
            if msgs and isinstance(msgs[-1], dict) and "value" in msgs[-1]:
                msgs[-1]["value"] += ("\n请根据图片完成描述性回答。回答必须围绕图中可见内容，说明关键对象、"
                                      "结构/位置关系、标注信息、数值或显著差异；不要只给通用功能解释。")
            return msgs

        def evaluate(self, eval_file, **judge_kwargs):
            data = load(eval_file)
            assert "answer" in data and "prediction" in data, "结果文件缺 answer/prediction 列"

            # 取图片给裁判看：优先预测文件里的 image 列；没有就回原始 TSV 按 index 取
            img_map = {}
            if "image" in data.columns and data["image"].notna().any():
                img_map = {r["index"]: r["image"] for _, r in data.iterrows()
                           if r.get("image") not in (None, "")}
            else:
                try:
                    ds = getattr(self, "dataset_name", None) or getattr(self, "dataset", "aero_vqa")
                    lmu = os.path.expanduser(os.environ.get("LMUData", "~/LMUData"))
                    orig = load(os.path.join(lmu, f"{ds}.tsv"))
                    if "image" in orig.columns:
                        img_map = {r["index"]: r["image"] for _, r in orig.iterrows()}
                        print(f"[aero_vqa] 预测文件无 image 列，已从原始 TSV 取回 {len(img_map)} 张图给裁判看图。")
                except Exception as e:
                    print(f"[aero_vqa] 警告：拿不到图片，裁判将退回纯文本判分（不理想）：{e}")

            scored, metrics = score_dataframe(data, img_map=img_map)

            # 落盘：逐条分数 + 指标（与 VLMEvalKit 习惯一致，文件名挂在 eval_file 旁）
            base = eval_file.rsplit(".", 1)[0]
            dump(scored, base + "_judge_detail.xlsx")          # 逐条 hit + 理由，便于核查
            with open(base + "_aero_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)

            # 返回指标 DataFrame（VLMEvalKit 期望 evaluate 返回一个指标表）
            import pandas as pd
            flat = {"Overall": metrics["overall"], "N": metrics["n"],
                    "BLEU-1": metrics["bleu1"], "BLEU-2": metrics["bleu2"],
                    "ROUGE-L": metrics["rouge_l"], "AvgPredLen": metrics["avg_pred_len"]}
            for k, v in (metrics.get("by_capability") or {}).items():
                flat[f"cap:{k}"] = v
            for k, v in (metrics.get("by_content_type") or {}).items():
                flat[f"ct:{k}"] = v
            ret = pd.DataFrame([flat])
            dump(ret, base + "_acc.csv")
            return ret

    return AeroVQADataset


# 模块级别名：import 时尝试构建；vlmeval 未装则留 None（不影响单测打分逻辑）
try:
    AeroVQADataset = _build_class()
except Exception as _e:  # noqa
    AeroVQADataset = None
    _IMPORT_ERR = str(_e)


if __name__ == "__main__":
    print("本文件提供 AeroVQADataset（VLMEvalKit 自定义 VQA 数据集，LLM 裁判打分）。")
    print("打分核心函数 score_dataframe 可独立测试。")
    if AeroVQADataset is None:
        print("注意：当前环境未装 vlmeval，数据集类未构建（仅打分逻辑可用）。")
