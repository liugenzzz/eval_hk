"""C / E / F 三组打分器 + 拒答表。"""

import math

import pandas as pd
import pytest

from eval_tool import scorers


def _score(kind, rows, **params):
    ctx = scorers.ScoringContext(dataset_key=kind, kind=kind, model_name="m", params=params)
    return scorers.get(kind).score(pd.DataFrame(rows), ctx)


# ------------------------------------------------------------- E 组物体识别

def _reg(label, prediction, index="1"):
    return {"index": index, "meta.label": label, "answer": f"该区域内的是{label}。",
            "prediction": prediction}


@pytest.fixture
def class_table(tmp_path):
    """正式跑必须配类别表：兜底表只认评估集里出现过的 label，模型答一个别的合法
    类别会被记成 off_table 多走一次裁判。"""
    path = tmp_path / "classes.yaml"
    path.write_text(
        "names:\n" + "".join(f"  - {name}\n" for name in
                             ["人员", "三轮车", "遮阳三轮车", "卡车", "货车", "切管器", "切管机"]),
        encoding="utf-8",
    )
    return str(path)


def test_exact_hit_is_the_only_thing_that_counts_as_hit(class_table):
    out = _score("object_ident", [
        _reg("遮阳三轮车", "该区域内的是遮阳三轮车。", "1"),
        _reg("遮阳三轮车", "这是三轮车", "2"),
        _reg("三轮车", "这是遮阳三轮车", "3"),
        _reg("三轮车", "这是卡车", "4"),
    ], classes_yaml=class_table)
    assert out["hit"].tolist() == [1, 0, 0, 0]
    assert out["match_kind"].tolist() == ["exact", "hypernym", "hyponym", "other"]


def test_hypernym_and_hyponym_hits_are_reported_separately_never_merged():
    """合成会掩盖一个真实问题：模型可能学会了「答粗一点更安全」，那是退化不是能力。"""
    out = _score("object_ident", [
        _reg("遮阳三轮车", "这是三轮车", "1"),
        _reg("三轮车", "这是遮阳三轮车", "2"),
    ])
    assert out["hit_hypernym"].tolist() == [1, 0]
    assert out["hit_hyponym"].tolist() == [0, 1]
    assert out["hit"].sum() == 0


def test_off_table_answer_is_queued_for_the_judge_not_scored_wrong_silently():
    out = _score("object_ident", [_reg("三轮车", "这是一台小电驴")])
    assert out.loc[0, "match_kind"] == "off_table"
    assert out.loc[0, "judge_fallback_needed"] == 1


def test_confusable_errors_are_separated_from_unrelated_ones(class_table):
    """错在易混组内补的是细粒度区分的数据，错得毫无关系是模型压根没认出来。"""
    rows = [
        {"index": "1", "meta.label": "切管器", "answer": "是切管器。", "prediction": "是切管机"},
        {"index": "2", "meta.label": "切管器", "answer": "是切管器。", "prediction": "是人员"},
    ]
    out = _score("object_ident", rows, classes_yaml=class_table)
    assert out["error_confusable"].tolist() == [1, 0]


def test_class_table_from_params_lets_the_model_name_a_class_not_in_this_eval_set(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text("names:\n  - 三轮车\n  - 遮阳三轮车\n  - 卡车\n", encoding="utf-8")
    out = _score("object_ident", [_reg("三轮车", "这是卡车")], classes_yaml=str(path))
    # 兜底表里只有「三轮车」，配了类别表之后「卡车」才认得出来，不会误判成 off_table。
    assert out.loc[0, "match_kind"] == "other"


def test_object_ident_without_a_class_table_or_label_column_says_what_is_missing():
    with pytest.raises(ValueError, match="classes_yaml"):
        _score("object_ident", [{"index": "1", "answer": "是卡车。", "prediction": "是卡车"}])


# ------------------------------------------------------------- C 组短答案

def test_short_answer_exact_match_ignores_punctuation():
    out = _score("short_answer", [
        {"index": "1", "answer": "白色车身。", "prediction": "白色车身"},
        {"index": "2", "answer": "白色车身。", "prediction": "银灰色"},
    ])
    assert out["hit"].tolist() == [1, 0]
    assert out.loc[1, "judge_fallback_needed"] == 1


def test_loose_match_is_a_diagnostic_column_not_the_metric():
    """「白色车身的面包车」包含「白色车身」，宽松口径会把多说了别的东西的答案算对。"""
    out = _score("short_answer", [{"index": "1", "answer": "白色车身", "prediction": "白色车身的面包车"}])
    assert out.loc[0, "hit"] == 0
    assert out.loc[0, "hit_loose"] == 1


# ------------------------------------------------------------- F 组计数

def _count(gold, prediction, counting="exact", index="1"):
    return {"index": index, "meta.count": gold, "meta.counting": counting,
            "answer": str(gold), "prediction": prediction}


def test_counting_reports_accuracy_error_and_direction_together():
    """准确率、MAE、偏向三个数缺一个都不够：只有符号说得清模型是往多了猜还是往少了猜，
    而这两种要补的数据完全不同。"""
    out = _score("counting", [
        _count(3, "图中一共有 3 辆卡车。", index="1"),
        _count(7, "大约有 5 辆。", index="2"),
        _count(2, "有 4 辆。", index="3"),
    ])
    assert out["hit"].tolist() == [1, 0, 0]
    assert out["count_abs_error"].tolist() == [0, 2, 2]
    assert out["count_error"].tolist() == [0, -2, 2]      # 一个漏报一个多报，平均抵消
    assert out["count_error"].mean() == 0


def test_gold_count_comes_from_metadata_not_recounted():
    """构建期已经做过跨任务一致性核对，评估端另算一份就等于配了两套真值。"""
    row = _count(3, "有 3 辆卡车")
    row["answer"] = "有 99 辆卡车"     # 答案文本故意和 metadata 不一致
    out = _score("counting", [row])
    assert out.loc[0, "gold_count"] == 3
    assert out.loc[0, "hit"] == 1


def test_unparseable_count_is_flagged_and_kept_out_of_the_accuracy():
    """解析不出来的整条记格式不合规，不计入准确率 —— 否则解析器的脆弱会被算成模型的错。"""
    out = _score("counting", [_count(2, "不好说")])
    assert out.loc[0, "parse_ok"] is False or out.loc[0, "parse_ok"] == False  # noqa: E712
    assert pd.isna(out.loc[0, "hit"])
    assert out.loc[0, "fail_reason"] == "unparseable_prediction"


def test_count_bins_and_question_styles_ride_along_as_columns():
    out = _score("counting", [
        _count(1, "1 辆", index="1"),
        _count(4, "4 辆", "visible", index="2"),
        _count(9, "9 辆", index="3"),
    ])
    assert out["count_bin"].tolist() == ["单例", "少量", "密集"]
    assert out["counting"].tolist() == ["exact", "visible", "exact"]


def test_the_zero_route_leaves_the_counting_accuracy_and_joins_the_refusal_table():
    """counting == "zero" 和 exist_negative 考的是同一件事（会不会「被问就一定有」），
    只是输出形态从「没有」换成「0」。两种形态在拒答表里要分行列。"""
    out = _score("counting", [
        _count(0, "图中没有货车。", "zero", index="1"),
        _count(0, "有 1 辆货车。", "zero", index="2"),
    ])
    assert out["hit"].isna().all()                      # 不进 F 组准确率
    assert out["refusal_hit"].tolist() == [1, 0]
    assert out["said_yes"].tolist() == [0, 1]
    assert out["refusal_form"].tolist() == ["count_zero", "count_zero"]


# ------------------------------------------------------------- F 组清单

def _inv(prediction, index="1", gold=("人员x3", "卡车x2")):
    return {"index": index, "meta.inventory": list(gold), "answer": "有3名人员、2辆卡车。",
            "prediction": prediction}


def test_inventory_splits_the_label_set_from_the_counts():
    """类别报全了但数全错，和类别漏一半但数都对，是完全不同的问题。"""
    out = _score("inventory", [
        _inv("有3名人员、2辆卡车。", "1"),
        _inv("有3名人员、5辆卡车。", "2"),   # 类别全对，数错一个
        _inv("有3名人员。", "3"),           # 漏一个类别，数全对
    ])
    assert out["set_f1"].tolist() == [1.0, 1.0, pytest.approx(2 / 3)]
    assert out["count_hit_rate"].tolist() == [1.0, 0.5, 1.0]


def test_inventory_counts_are_measured_only_on_matched_labels():
    out = _score("inventory", [_inv("有3名人员、9艘船。", "1")])
    assert out.loc[0, "labels_missed"] == "卡车"
    assert out.loc[0, "labels_spurious"] == "船"
    assert out.loc[0, "count_hit_rate"] == 1.0     # 对上的只有人员，那一项数对了


def test_unparseable_inventory_still_records_the_missed_labels():
    out = _score("inventory", [_inv("这张图看起来很复杂", "1")])
    assert out.loc[0, "parse_ok"] is False or out.loc[0, "parse_ok"] == False  # noqa: E712
    assert out.loc[0, "set_recall"] == 0.0
    assert math.isnan(out.loc[0, "count_hit_rate"])


# ------------------------------------------------------------- 拒答表

def test_exist_negative_reports_accuracy_and_yes_bias():
    """SFT 之后 yes 偏置变高是常见退化，而它在总准确率里是看不见的。"""
    out = _score("exist_negative", [
        {"index": "1", "meta.polarity": "negative", "answer": "没有。", "prediction": "图中没有货车。"},
        {"index": "2", "meta.polarity": "negative", "answer": "没有。", "prediction": "有，图中有 1 辆货车。"},
        {"index": "3", "meta.polarity": "positive", "answer": "有。", "prediction": "有，图中有 2 辆自行车。"},
    ])
    assert out["hit"].tolist() == [1, 0, 1]
    assert out["said_yes"].tolist() == [0, 1, 1]
    # 拒答准确率只在负样本上算
    assert out["refusal_hit"].dropna().tolist() == [1, 0]


def test_negation_is_checked_before_affirmation():
    """「没有」里含「有」，判定顺序反了会把每一条都判成 yes。"""
    out = _score("exist_negative", [
        {"index": "1", "meta.polarity": "negative", "answer": "没有。", "prediction": "没有"},
    ])
    assert out.loc[0, "said_yes"] == 0
