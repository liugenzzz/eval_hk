"""§11.2 问法扰动打分器 + §8.2 链路衰减率。"""

import math

import pandas as pd
import pytest

from eval_tool import scorers
from eval_tool.breakdown import make_chain_decay

GOLD = '{"bbox_2d":[100,100,200,200]}'


def _group(group_id, predictions):
    return [
        {"index": f"{group_id}__pt{i}", "model": "sft", "dataset": "pt",
         "meta.perturb_group": group_id, "task_type": "question_perturbation",
         "question": "q", "answer": GOLD, "prediction": prediction}
        for i, prediction in enumerate(predictions)
    ]


def _score(rows, **params):
    ctx = scorers.ScoringContext(dataset_key="pt", kind="perturbation",
                                model_name="sft", params=params)
    return scorers.get("perturbation").score(pd.DataFrame(rows), ctx)


def test_one_row_per_group_not_per_variant():
    """一致性是组的属性。按变体行平均等于把同一个组数了三遍。"""
    rows = _group("a", ["[100,100,200,200]"] * 3) + _group("b", ["[100,100,200,200]"] * 3)
    out = _score(rows)
    assert len(out) == 2
    assert sorted(out["index"]) == ["a", "b"]
    assert out["n_variants"].tolist() == [3, 3]


def test_stable_answers_across_phrasings_count_as_consistent():
    out = _score(_group("a", ["[100,100,200,200]", "[101,100,199,201]", "[100,101,200,200]"]))
    assert out.loc[0, "all_pairs_consistent"] == 1
    assert out.loc[0, "hit"] == 1
    assert out.loc[0, "pairwise_iou_min"] >= 0.9


def test_one_phrasing_that_breaks_the_model_fails_the_whole_group():
    """SFT 数据的问法有限，模型很容易学成「看到某个固定句式才输出坐标」，
    换个说法就崩 —— 三种说法里崩一种，这一组就不算稳。"""
    out = _score(_group("a", ["[100,100,200,200]", "[400,400,500,500]", "[100,100,200,200]"]))
    assert out.loc[0, "all_pairs_consistent"] == 0
    assert out.loc[0, "pairwise_iou_min"] == 0.0
    assert out.loc[0, "coord_std_mean"] > 100


def test_coordinate_spread_is_reported_per_point():
    out = _score(_group("a", ["[100,100,200,200]", "[120,100,200,200]", "[140,100,200,200]"]))
    assert out.loc[0, "coord_std_x1"] > 0
    assert out.loc[0, "coord_std_y1"] == 0.0


def test_consistent_but_consistently_wrong_is_visible():
    """一致但一致地错，和一致且对，是完全不同的结论。"""
    out = _score(_group("a", ["[700,700,800,800]"] * 3))
    assert out.loc[0, "hit"] == 1                    # 三次输出彼此一致
    assert out.loc[0, "gold_iou_mean"] == 0.0        # 但都没框对
    assert out.loc[0, "dev_mean4_pct_mean"] > 0


def test_a_group_that_mostly_failed_to_parse_scores_na_not_zero():
    """「换个说法就不输出坐标了」是格式合规率的事，在这里记 0 等于罚两次。"""
    out = _score(_group("a", ["[100,100,200,200]", "没有找到", "也没有"]))
    assert pd.isna(out.loc[0, "hit"])
    assert out.loc[0, "fail_reason"] == "not_enough_parsed_variants"
    assert out.loc[0, "n_parsed"] == 1


def test_two_parsed_variants_are_enough_to_judge():
    out = _score(_group("a", ["[100,100,200,200]", "[100,100,200,200]", "没有找到"]))
    assert out.loc[0, "n_parsed"] == 2
    assert out.loc[0, "hit"] == 1


def test_the_consistency_gate_comes_from_params():
    rows = _group("a", ["[100,100,200,200]", "[110,110,210,210]", "[100,100,200,200]"])
    assert _score(rows, consistent_iou=0.5).loc[0, "hit"] == 1
    assert _score(rows, consistent_iou=0.95).loc[0, "hit"] == 0


def test_metadata_rides_along_for_the_report_dimensions():
    rows = _group("a", ["[100,100,200,200]"] * 3)
    for row in rows:
        row["meta.size_bucket"] = "small"
    out = _score(rows)
    assert out.loc[0, "meta.size_bucket"] == "small"


def test_a_dataset_without_a_group_column_is_refused():
    rows = [{"index": "1", "model": "m", "dataset": "pt", "answer": GOLD,
             "prediction": "[1,2,3,4]", "question": "q", "task_type": "x"}]
    with pytest.raises(ValueError, match="分组列"):
        _score(rows)


# ------------------------------------------------------------- 链路衰减率

def _chain_rows(gold_score, model_score, n=40):
    rows = []
    for i in range(n):
        rows.append({"model": "sft", "dataset": "describe", "sample_id": f"s{i}", "hit": gold_score})
        rows.append({"model": "sft", "dataset": "describe_mh", "sample_id": f"s{i}__mh",
                     "meta.derived_from": f"s{i}", "hit": model_score})
    return pd.DataFrame(rows)


def test_chain_decay_is_the_relative_drop_from_gold_history_to_model_history():
    out = make_chain_decay(_chain_rows(0.8, 0.6), [{"gold": "describe", "model": "describe_mh"}])
    row = out.iloc[0]
    assert row["gold_score"] == 0.8
    assert row["model_history_score"] == 0.6
    assert row["chain_decay"] == pytest.approx(0.25)
    assert row["status"] == "ok"


def test_no_decay_when_the_model_history_scores_the_same():
    out = make_chain_decay(_chain_rows(0.8, 0.8), [{"gold": "describe", "model": "describe_mh"}])
    assert out.iloc[0]["chain_decay"] == 0.0


def test_chain_decay_compares_the_same_samples_on_both_sides():
    """model 那一路只跑 30% 抽样。直接比两个均值等于拿不同的两批样本相减 ——
    抽样子集碰巧偏难，衰减率就凭空多出一截。"""
    rows = []
    for i in range(60):
        rows.append({"model": "sft", "dataset": "describe", "sample_id": f"s{i}",
                     "hit": 0.8 if i < 20 else 0.2})
    for i in range(20):
        rows.append({"model": "sft", "dataset": "describe_mh", "sample_id": f"s{i}__mh",
                     "meta.derived_from": f"s{i}", "hit": 0.6})
    out = make_chain_decay(pd.DataFrame(rows), [{"gold": "describe", "model": "describe_mh"}],
                           min_n=10).iloc[0]
    assert out["gold_score"] == 0.8          # 只用抽中的那 20 条，不是全部 60 条
    assert out["n_gold"] == 20
    assert out["chain_decay"] == pytest.approx(0.25)


def test_small_samples_are_marked_rather_than_reported_as_a_conclusion():
    """衰减率是两个均值相除，n 小的时候它比任何一个均值都不稳。"""
    out = make_chain_decay(_chain_rows(0.8, 0.6, n=5), [{"gold": "describe", "model": "describe_mh"}])
    assert out.iloc[0]["status"] == "insufficient"


def test_a_missing_pair_yields_no_row():
    assert make_chain_decay(_chain_rows(0.8, 0.6),
                            [{"gold": "describe", "model": "nope"}]).empty
