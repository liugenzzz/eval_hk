"""§17 报表拆分：多维交叉表、标灰规则、三种空格子、加权总分。

一个总分没有用。这次评估的真正产出是「哪一档、哪一类、哪种尺寸的目标还不行」，
所有设计都服务于**能拆开看**。

标灰规则（§12.4）
-----------------
``n`` 太小的格子给不出结论，但也不能删掉 —— 删掉了读表的人不知道那里试过。
所以每一格都带一个 ``status``：

======================  =========================================================
``ok``                   n 够，可以下结论
``trend_only``           维度声明了 ``level="group"``（如难度档在 task 级），只看趋势
``insufficient``         n < min_n，显示 ``n=xx（样本不足）``，不显示百分比
``by_design``            该组合**本就不产样本**（ground_part 没有 hard 档）
``not_in_data``          该任务**产出为 0**（ground_unique / spatial_relation）
======================  =========================================================

后三种在报表上都是空白。**实现的人看到空格会当 bug 修**，所以必须分开标注：
「设计上为空」「训练集中不存在」「样本不足」是三件完全不同的事。

加权总分
--------
只由 ``engine="code"`` 的数据集构成。裁判打的分会抖，做验收不合适（§15.4 也写了
D 组只能纵向对比不能当绝对值报）。代码打分器保证同一份预测重跑一百遍逐位相同，
一个会抖的总分没法用来验收。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .aggregate import bootstrap_mean_ci

MIN_N_FOR_CONCLUSION = 30

OK = "ok"
TREND_ONLY = "trend_only"
INSUFFICIENT = "insufficient"
BY_DESIGN = "by_design"
NOT_IN_DATA = "not_in_data"

TASK_LEVEL = "task"
GROUP_LEVEL = "group"


@dataclass(frozen=True)
class DimSpec:
    """一根报表维度。``source`` 是取值所在的列，其余都是怎么呈现的事。"""

    key: str
    source: str
    level: str = TASK_LEVEL
    min_n: int = MIN_N_FOR_CONCLUSION
    only_kinds: tuple[str, ...] = ()
    worst_n: int | None = None
    bins: tuple[tuple[int, int | None, str], ...] = ()

    def value_of(self, raw: Any) -> str:
        if not self.bins:
            return "" if raw is None or _is_na(raw) else str(raw)
        try:
            number = int(raw)
        except (TypeError, ValueError):
            return ""
        for low, high, label in self.bins:
            if number >= low and (high is None or number <= high):
                return label
        return ""


def _is_na(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def parse_dims(raw: Sequence[Mapping[str, Any]] | None) -> list[DimSpec]:
    """把配置里的 report.dims 解析成 DimSpec。维度名不写死在代码里 —— 换一套数据
    只改配置，不改报表代码。"""
    dims: list[DimSpec] = []
    for item in raw or []:
        key = str(item.get("key") or item.get("from") or "").strip()
        source = str(item.get("from") or key).strip()
        if not key or not source:
            raise ValueError(f"report.dims 里有一项缺 key/from：{item}")
        bins_raw = item.get("bins") or []
        bins = tuple(
            (int(entry[0]), None if entry[1] is None else int(entry[1]), str(entry[2]))
            for entry in bins_raw
        )
        only = item.get("only_kinds") or ()
        dims.append(
            DimSpec(
                key=key,
                source=source,
                level=str(item.get("level") or TASK_LEVEL),
                min_n=int(item.get("min_n", MIN_N_FOR_CONCLUSION)),
                only_kinds=tuple(str(k) for k in ([only] if isinstance(only, str) else only)),
                worst_n=int(item["worst_n"]) if item.get("worst_n") else None,
                bins=bins,
            )
        )
    return dims


@dataclass(frozen=True)
class EmptyCells:
    """配置里显式声明的空格子。不声明的话报表上就是一片空白，看不出是哪种空。"""

    by_design: tuple[tuple[str, ...], ...] = ()
    not_in_data: tuple[str, ...] = ()

    @classmethod
    def parse(cls, raw: Mapping[str, Any] | None) -> "EmptyCells":
        raw = raw or {}
        return cls(
            by_design=tuple(tuple(str(v) for v in entry) for entry in raw.get("by_design") or ()),
            not_in_data=tuple(str(v) for v in raw.get("not_in_data") or ()),
        )


def make_breakdown(
    details: pd.DataFrame,
    dims: Sequence[DimSpec],
    metrics: Sequence[str] = ("hit",),
    *,
    kinds: Mapping[str, str] | None = None,
    empty_cells: EmptyCells | None = None,
    bootstrap_n: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """长表：每行是「模型 × 数据集 × 维度 × 取值 × 指标」的一格，带 n / CI / status。"""
    if details.empty or "model" not in details.columns:
        return pd.DataFrame()
    kinds = kinds or {}
    empty_cells = empty_cells or EmptyCells()
    rows: list[dict[str, Any]] = []

    for (model, dataset), frame in details.groupby(["model", "dataset"], dropna=False):
        kind = str(kinds.get(str(dataset), ""))
        for dim in dims:
            if dim.only_kinds and kind and kind not in dim.only_kinds:
                continue
            if dim.source not in frame.columns:
                continue
            values = frame[dim.source].map(dim.value_of)
            for metric in metrics:
                if metric not in frame.columns:
                    continue
                cells = _cells_for(frame, values, metric, dim, bootstrap_n, seed)
                if dim.worst_n:
                    cells = sorted(
                        cells, key=lambda c: (math.inf if _is_na(c["score"]) else c["score"])
                    )[: dim.worst_n]
                for cell in cells:
                    cell.update({"model": str(model), "dataset": str(dataset), "dim": dim.key,
                                 "metric": metric})
                    rows.append(cell)

        for entry in empty_cells.by_design:
            rows.append(_declared_empty(model, dataset, entry, BY_DESIGN, dims))
        for task in empty_cells.not_in_data:
            rows.append(_declared_empty(model, dataset, (task,), NOT_IN_DATA, dims))

    columns = ["model", "dataset", "dim", "value", "metric", "score", "n", "ci_low", "ci_high", "status"]
    out = pd.DataFrame([{col: row.get(col) for col in columns} for row in rows])
    return out[columns] if not out.empty else out


def _cells_for(
    frame: pd.DataFrame,
    values: pd.Series,
    metric: str,
    dim: DimSpec,
    bootstrap_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    numeric = pd.to_numeric(frame[metric], errors="coerce")
    for value, index in values.groupby(values).groups.items():
        if str(value) == "":
            continue
        scores = numeric.loc[index].dropna()
        n = int(len(scores))
        if n == 0:
            # 这一格一个可用值都没有（比如关了 pointwise 时 describe 的 hit 全是 NA，
            # 或者 counting=zero 那一路按设计不进 F 组准确率）。这不是「样本不足」，
            # 是这个指标对这一格不适用 —— 照样出行只会在报表里刷几十条无信息的灰格。
            continue
        if n < dim.min_n:
            # n 太小的格子显示 n，但不显示百分比 —— 一个 n=3 的 100% 会被当成结论。
            status, score, low, high = INSUFFICIENT, math.nan, math.nan, math.nan
        else:
            status = TREND_ONLY if dim.level == GROUP_LEVEL else OK
            score = round(float(scores.mean()), 4)
            low, high = bootstrap_mean_ci(scores, n_bootstrap=bootstrap_n, seed=seed)
        cells.append({"value": str(value), "score": score, "n": n,
                      "ci_low": low, "ci_high": high, "status": status})
    return cells


def _declared_empty(
    model: Any, dataset: Any, entry: Sequence[str], status: str, dims: Sequence[DimSpec]
) -> dict[str, Any]:
    dim_key = dims[len(entry) - 1].key if len(entry) <= len(dims) else "cell"
    return {
        "model": str(model), "dataset": str(dataset), "dim": dim_key,
        "value": " × ".join(entry), "metric": "", "score": math.nan, "n": 0,
        "ci_low": math.nan, "ci_high": math.nan, "status": status,
    }


def make_failure_buckets(details: pd.DataFrame) -> pd.DataFrame:
    """达标率 + 三个失败桶。加起来必须等于 1，否则桶定义漏了一种情形。

    只报一个 75% 没法指导补数据：格式不合规是训练配置的问题，定位失败要补指代
    消歧和密集场景，偏差超标要补精细边界（再看 bias_* 定方向）。
    """
    required = {"model", "dataset", "fail_bucket"}
    if details.empty or not required.issubset(details.columns):
        return pd.DataFrame()
    frame = details[details["fail_bucket"].notna()]
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (model, dataset), group in frame.groupby(["model", "dataset"], dropna=False):
        buckets = group["fail_bucket"].astype(str)
        scored = buckets[buckets != "gt_unparseable"]
        n = int(len(scored))
        if not n:
            continue
        row: dict[str, Any] = {
            "model": str(model), "dataset": str(dataset), "n": n,
            "pass_rate": round(float((scored == "").mean()), 4),
        }
        for bucket in ("malformed", "localize_fail", "deviation"):
            row[f"fail_{bucket}"] = round(float((scored == bucket).mean()), 4)
        row["gt_unparseable_n"] = int((buckets == "gt_unparseable").sum())
        rows.append(row)
    return pd.DataFrame(rows)


def make_weighted_total(
    details: pd.DataFrame,
    weights: Mapping[str, float],
    engines: Mapping[str, str],
    metric_col: str = "hit",
    min_n: int = MIN_N_FOR_CONCLUSION,
) -> pd.DataFrame:
    """验收总分：只由 engine="code" 的数据集构成，按 dataset 权重加权。

    裁判打的分不进总分。它会抖，做验收不合适 —— 同一份预测重跑两遍数字就不一样，
    而验收需要的是「重跑一百遍逐位相同」。D 组的结论只做纵向对比，单独成表。
    """
    if details.empty or metric_col not in details.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model, frame in details.groupby("model"):
        per_dataset: dict[str, float] = {}
        counts: dict[str, int] = {}
        skipped_judge: list[str] = []
        skipped_small: list[str] = []
        for dataset, group in frame.groupby("dataset"):
            key = str(dataset)
            if engines.get(key) != "code":
                skipped_judge.append(key)
                continue
            values = pd.to_numeric(group[metric_col], errors="coerce").dropna()
            if len(values) < min_n:
                skipped_small.append(f"{key}(n={len(values)})")
                continue
            per_dataset[key] = round(float(values.mean()), 4)
            counts[key] = int(len(values))
        weight_sum = sum(float(weights.get(k, 1.0)) for k in per_dataset)
        total = (
            round(sum(float(weights.get(k, 1.0)) * v for k, v in per_dataset.items()) / weight_sum, 4)
            if weight_sum
            else math.nan
        )
        row: dict[str, Any] = {
            "model": str(model),
            "total_score": total,
            "n_total": int(sum(counts.values())),
            "datasets_in_total": ",".join(sorted(per_dataset)),
            "excluded_judge_datasets": ",".join(sorted(skipped_judge)),
            "excluded_small_n": ",".join(sorted(skipped_small)),
        }
        for dataset in sorted(per_dataset):
            row[dataset] = per_dataset[dataset]
            row[f"{dataset}_n"] = counts[dataset]
            row[f"{dataset}_weight"] = float(weights.get(dataset, 1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def default_dims() -> list[DimSpec]:
    """目标检测评估集的默认维度。配置里写了 report.dims 就以配置为准。"""
    return parse_dims(
        [
            {"key": "task_type", "from": "task_type"},
            # 难度档只在组级汇总下结论：task × difficulty 每格 n 只有几十，
            # 最坏 95% CI 半宽 ±13.9pp，每一格都得标灰。
            {"key": "difficulty", "from": "meta.difficulty", "level": GROUP_LEVEL},
            {"key": "size_bucket", "from": "meta.size_bucket",
             "only_kinds": ["grounding_single", "grounding_multi"]},
            {"key": "label", "from": "meta.label", "worst_n": 20},
            # 7 种 describe kind 的答案信息结构完全不同，合成一个平均分等于把这次
            # 数据集设计的核心抹掉。报表必须有 7 行。
            {"key": "describe_kind", "from": "meta.describe_kind",
             "only_kinds": ["judge_text", "describe"]},
            # §2.1：D 组内部按上游形态拆行 —— 文字指代 / 坐标回指 / 上文承接是三种
            # 不同的能力，平均成一个数就看不出是哪一种垮了。
            {"key": "upstream_form", "from": "upstream_form", "only_kinds": ["describe"]},
            {"key": "count_bin", "from": "meta.count", "only_kinds": ["counting"],
             "bins": [[1, 1, "单例"], [2, 5, "少量"], [6, None, "密集"]]},
            {"key": "counting", "from": "meta.counting", "only_kinds": ["counting"]},
            # region_identify 既有「说出类别」也有「用一个词回答」（answer_format=short）。
            # 短答案那一路模型更容易只吐一个词，混在一起报会把这个混淆因素算进能力差异。
            {"key": "answer_format", "from": "meta.answer_format"},
            # 拒答表按正负样本和 hard negative 拆。负样本是易混类别（图里有卡车问货车），
            # 相当于 POPE 的 Adversarial 档；数据里若同时有 hard_negative=false 的负样本，
            # 两档的差值就免费给出了「模型是在看图还是在用语言先验猜」的对照。
            {"key": "polarity", "from": "meta.polarity",
             "only_kinds": ["exist_negative", "counting"]},
            {"key": "hard_negative", "from": "meta.hard_negative",
             "only_kinds": ["exist_negative", "counting"]},
        ]
    )


def make_chain_decay(
    details: pd.DataFrame,
    pairs: Sequence[Mapping[str, str]],
    metric_col: str = "hit",
    min_n: int = MIN_N_FOR_CONCLUSION,
) -> pd.DataFrame:
    """§8.2 链路衰减率 = (D_gold 得分 − D_model 得分) / D_gold 得分。

    这是「专项训练该补哪边」的直接依据：

    ==================  ====================================================
    衰减大               瓶颈在**定位** —— 框错了描述再好也白搭，补定位数据
    衰减小但 D_gold 低    瓶颈在**描述** —— 补描述数据
    两个都高             这一轮 SFT 成了
    ==================  ====================================================

    ``pairs`` 形如 ``[{"gold": "describe", "model": "describe_modelhist"}]``。
    model 那一路只跑 30% 抽样，所以两边的 n 不一样是正常的；但**要在同一批样本上比**，
    所以这里按 ``derived_from`` 对齐，只用两边都有的那些样本算。
    """
    if details.empty or metric_col not in details.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        gold_key, model_key = str(pair.get("gold", "")), str(pair.get("model", ""))
        gold_all = details[details["dataset"] == gold_key]
        model_all = details[details["dataset"] == model_key]
        if gold_all.empty or model_all.empty:
            continue
        for model_name in sorted(set(gold_all["model"]) & set(model_all["model"])):
            gold = gold_all[gold_all["model"] == model_name]
            model_hist = model_all[model_all["model"] == model_name]
            gold, model_hist = _align_on_source(gold, model_hist)
            gold_values = pd.to_numeric(gold[metric_col], errors="coerce").dropna()
            model_values = pd.to_numeric(model_hist[metric_col], errors="coerce").dropna()
            if not len(gold_values) or not len(model_values):
                continue
            gold_score = float(gold_values.mean())
            model_score = float(model_values.mean())
            decay = (gold_score - model_score) / gold_score if gold_score else math.nan
            rows.append({
                "model": model_name,
                "gold_dataset": gold_key,
                "model_dataset": model_key,
                "metric": metric_col,
                "gold_score": round(gold_score, 4),
                "model_history_score": round(model_score, 4),
                "chain_decay": round(decay, 4) if not math.isnan(decay) else math.nan,
                "n_gold": int(len(gold_values)),
                "n_model_history": int(len(model_values)),
                # n 太小就别下结论 —— 衰减率是两个均值相除，n 小的时候它比任何一个
                # 均值都不稳。
                "status": OK if min(len(gold_values), len(model_values)) >= min_n else INSUFFICIENT,
            })
    return pd.DataFrame(rows)


def _align_on_source(gold: pd.DataFrame, model_hist: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把两边收敛到同一批原始样本上。

    model 历史那一路只跑 30% 抽样，直接比两个均值等于拿不同的两批样本相减 —— 抽样
    子集碰巧偏难，衰减率就凭空多出一截。派生集的每条都带 ``meta.derived_from``
    指回原始记录 id，主线那边是 ``sample_id``，按这两列取交集。

    对不上（缺列）时原样返回：这时候的衰减率只是个粗略值，n 会如实报出来。
    """
    source_col = next((c for c in ("meta.derived_from", "derived_from") if c in model_hist.columns), None)
    gold_col = "sample_id" if "sample_id" in gold.columns else None
    if source_col is None or gold_col is None:
        return gold, model_hist
    shared = set(model_hist[source_col].dropna().astype(str)) & set(gold[gold_col].dropna().astype(str))
    if not shared:
        return gold, model_hist
    return (
        gold[gold[gold_col].astype(str).isin(shared)],
        model_hist[model_hist[source_col].astype(str).isin(shared)],
    )


CROSS_HIT_PREFIX = "hit__"
"""交叉裁判的分数列前缀。``hit__internvl`` = 名为 internvl 的裁判给的 hit。"""


def cross_judge_names(details: pd.DataFrame) -> list[str]:
    """明细表里有哪几路交叉裁判。"""
    return sorted(
        column[len(CROSS_HIT_PREFIX):]
        for column in details.columns
        if column.startswith(CROSS_HIT_PREFIX) and len(column) > len(CROSS_HIT_PREFIX)
    )


def _spearman(left: pd.Series, right: pd.Series) -> float:
    """秩相关。自己实现是因为 pandas 的 method="spearman" 要 scipy，而这个工具
    刻意不依赖 scipy（匈牙利匹配也是自己写的）。并列取平均秩。"""
    if len(left) < 3:
        return math.nan
    a = left.rank(method="average")
    b = right.rank(method="average")
    a_dev, b_dev = a - a.mean(), b - b.mean()
    denom = math.sqrt(float((a_dev**2).sum()) * float((b_dev**2).sum()))
    # 有一边全是同一个分（裁判把整批判成了 1）时秩没有方差，相关系数没有定义。
    # 报 nan 而不是 0：0 会被读成「两个裁判毫不相关」，而事实是这批数据分不出来。
    return float((a_dev * b_dev).sum() / denom) if denom else math.nan


def _paired_hits(frame: pd.DataFrame, cross_column: str) -> tuple[pd.Series, pd.Series]:
    """取两个裁判都打出分的那些行。判失败记的是 None，两边的失败行不一定重合，
    不取交集就等于拿不同的两批样本比裁判。"""
    primary = pd.to_numeric(frame.get("hit"), errors="coerce")
    cross = pd.to_numeric(frame.get(cross_column), errors="coerce")
    both = primary.notna() & cross.notna()
    return primary[both], cross[both]


def make_judge_agreement(
    details: pd.DataFrame, min_n: int = MIN_N_FOR_CONCLUSION
) -> pd.DataFrame:
    """§15.2 自偏检测：主裁判和异家族裁判逐行比。

    需求文档自己写了这一条做不到 ——「裁判是 Qwen3.8-27B，被测是 Qwen3-VL-8B，同家族，
    当前无法做自偏检测」。同家族裁判可能偏爱同家族的输出风格，而这件事在单裁判下
    **看不出来**：分高到底是模型强，还是裁判认亲，一个裁判给不出区分。

    配了 ``judge.cross_check`` 之后每行多一列 ``hit__<名字>``，这张表把它和主裁判
    的 hit 逐行对齐：

    ==================  ====================================================
    ``delta``            异家族裁判的均分 − 主裁判的均分。大正数 = 主裁判压分，
                         大负数 = 主裁判送分（同家族偏爱的典型形状）
    ``mad``              逐行绝对差的均值。均分相同但 mad 大 = 两个裁判在不同的
                         样本上给分，只是碰巧抵消了
    ``agree_rate``       以 0.5 为界的结论一致率
    ``spearman``         秩相关。它低而 delta 小最危险 —— 两边排序都不一样了
    ==================  ====================================================

    注意这张表只诊断裁判，不进验收总分：验收分只由 ``engine="code"`` 的数据集构成。
    """
    names = cross_judge_names(details) if not details.empty else []
    if not names or "hit" not in details.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for name in names:
        column = f"{CROSS_HIT_PREFIX}{name}"
        scoped = details[details[column].notna()]
        if scoped.empty:
            continue
        for (dataset, model), frame in scoped.groupby(["dataset", "model"], sort=True):
            primary, cross = _paired_hits(frame, column)
            n = int(len(primary))
            if not n:
                continue
            diff = (cross - primary).abs()
            agree = ((primary >= 0.5) == (cross >= 0.5)).mean()
            rows.append({
                "dataset": str(dataset),
                "model": str(model),
                "judge": name,
                "n": n,
                "mean_primary": round(float(primary.mean()), 4),
                "mean_cross": round(float(cross.mean()), 4),
                "delta": round(float(cross.mean() - primary.mean()), 4),
                "mad": round(float(diff.mean()), 4),
                "agree_rate": round(float(agree), 4),
                "spearman": round(_spearman(primary, cross), 4),
                "status": OK if n >= min_n else INSUFFICIENT,
            })
    return pd.DataFrame(rows)


def make_judge_conclusion(
    details: pd.DataFrame, baseline_model: str, min_n: int = MIN_N_FOR_CONCLUSION
) -> pd.DataFrame:
    """交叉裁判会不会推翻结论 —— 这才是自偏检测真正要回答的那一句。

    逐行一致率再难看，只要两个裁判都说「SFT 比 base 高」，结论就站得住；反过来，
    逐行一致率很高但**增益的符号翻了**，那份增益就不能报。所以这张表算的是同一个
    对比在两个裁判下各是多少：

    * ``agree``：两边增益同号（都不为 0 时），结论一致
    * ``flipped``：符号相反 —— 这份增益多半是裁判的家族偏好，不是能力差异
    * ``insufficient``：任一边样本不够，不下结论

    只跟 ``baseline_model`` 比。三个以上模型时两两比会把表撑爆，而验收要回答的
    始终是「相对基线涨了没有」。
    """
    names = cross_judge_names(details) if not details.empty else []
    if not names or "hit" not in details.columns or "model" not in details.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for name in names:
        column = f"{CROSS_HIT_PREFIX}{name}"
        scoped = details[details[column].notna()]
        if scoped.empty:
            continue
        for dataset in sorted(set(scoped["dataset"].astype(str))):
            frame = scoped[scoped["dataset"].astype(str) == dataset]
            base = frame[frame["model"].astype(str) == str(baseline_model)]
            if base.empty:
                continue
            base_primary, base_cross = _paired_hits(base, column)
            if not len(base_primary):
                continue
            for model in sorted(set(frame["model"].astype(str)) - {str(baseline_model)}):
                target = frame[frame["model"].astype(str) == model]
                primary, cross = _paired_hits(target, column)
                if not len(primary):
                    continue
                gain_primary = float(primary.mean() - base_primary.mean())
                gain_cross = float(cross.mean() - base_cross.mean())
                n = min(len(primary), len(base_primary))
                if n < min_n:
                    verdict = INSUFFICIENT
                elif gain_primary == 0.0 or gain_cross == 0.0:
                    verdict = "tie"
                elif (gain_primary > 0) == (gain_cross > 0):
                    verdict = "agree"
                else:
                    verdict = "flipped"
                rows.append({
                    "dataset": dataset,
                    "model": model,
                    "baseline": str(baseline_model),
                    "judge": name,
                    "gain_primary": round(gain_primary, 4),
                    "gain_cross": round(gain_cross, 4),
                    "n": int(n),
                    "verdict": verdict,
                })
    return pd.DataFrame(rows)
