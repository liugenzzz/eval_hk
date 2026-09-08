"""E 组物体识别：类别名比对，含上下位四档判定。纯代码，不调任何模型。

金标「遮阳三轮车」，模型答「三轮车」—— 算对还是错？分四档报（见
``eval_tool/classes``）：

- **精确命中率是主指标**，上位命中率、下位命中率各占一列，**不许合成**。
  合成会掩盖一个真实问题：模型可能学会了「答粗一点更安全」，那是退化不是能力；
  反过来「答细一点显得更专业」则是在幻觉一个它看不清的属性。
- 答案不在类别表里（模型自创了词）标成 ``off_table``，等裁判判一次是不是同义 ——
  这是 C / E 组唯一用到裁判的地方，兜底本身在阶段 6 接上。
- 错在易混组内（切管器 vs 切管机）和错得毫无关系分开计数：前者补细粒度区分的
  数据，后者是模型压根没认出来。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..classes import EXACT, HYPERNYM, HYPONYM, OFF_TABLE, OTHER, ClassTable, load_class_table, table_from_names
from . import CODE, ScoringContext, register, resolve_path


def class_table_for(ctx: ScoringContext, data: pd.DataFrame) -> ClassTable:
    """类别表优先从 ``params.classes_yaml`` 读；没配就用真值 label 列里出现过的类别兜底。

    兜底表只覆盖评估集里出现过的类别，判不出「模型答了一个别的合法类别」这种情形
    （会记成 off_table 多走一次裁判），所以正式跑必须配 classes_yaml。评估在内网机器
    上跑，那份表本来就在那儿，配一个路径即可，不需要把它搬进仓库。

    兜底只认 label 列，**不拿答案句子凑表**：「该区域内的是遮阳三轮车。」整句进表
    之后，最长匹配会把它当成一个类别名，模型原样复述反而被判成下位命中。
    """
    configured = ctx.params.get("classes_yaml") or ctx.params.get("classes_path")
    if configured:
        return load_class_table(resolve_path(ctx, configured))
    label_col = next((c for c in ("meta.label", "label") if c in data.columns), None)
    if label_col is None:
        raise ValueError(
            "object_ident 需要类别表：给 params 配 classes_yaml，或让数据带 meta.label 列"
        )
    names = {str(v).strip() for v in data[label_col].dropna().tolist() if str(v).strip()}
    return table_from_names(sorted(names))


def _gold_label(row: Any, table: ClassTable) -> str:
    """真值取 metadata 的 label；没有就从答案句子里抠（「该区域内的是面包车。」）。"""
    for column in ("meta.label", "label"):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    found = table.find_in_text(row.get("answer", ""))
    return found or str(row.get("answer", "") or "").strip()


@register("object_ident", engine=CODE)
def score_object_ident(data: pd.DataFrame, ctx: ScoringContext) -> pd.DataFrame:
    table = class_table_for(ctx, data)
    rows: list[dict[str, Any]] = []
    for _, row in data.iterrows():
        gold = _gold_label(row, table)
        prediction = row.get("prediction", "")
        pred_label = table.find_in_text(prediction)
        relation = table.relation(gold, pred_label) if pred_label else OFF_TABLE
        record = {
            "gold_label": gold,
            "pred_label": pred_label or "",
            "match_kind": relation,
            # 主指标：只有精确命中算对。
            "hit": int(relation == EXACT),
            "hit_hypernym": int(relation == HYPERNYM),
            "hit_hyponym": int(relation == HYPONYM),
            "off_table": int(relation == OFF_TABLE),
            # 裁判兜底的候选（约 5% 触发）。判词在阶段 6 接上，这里先把口子留出来。
            "judge_fallback_needed": int(relation == OFF_TABLE),
            "error_confusable": int(relation == OTHER and table.is_confusable(gold, pred_label)),
        }
        rows.append(record)
    scored = data.copy()
    for column in rows[0] if rows else []:
        scored[column] = [row[column] for row in rows]
    return scored
