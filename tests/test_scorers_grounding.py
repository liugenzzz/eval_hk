"""A 组单框 / B 组多框打分器。验收口径的每一条都要有测试守着。"""

import math

import pandas as pd
import pytest

from eval_tool import scorers


def _single(rows, **params):
    ctx = scorers.ScoringContext(
        dataset_key="ground", kind="grounding_single", model_name="m", params=params
    )
    return scorers.get("grounding_single").score(pd.DataFrame(rows), ctx)


def _multi(rows, **params):
    ctx = scorers.ScoringContext(
        dataset_key="detect", kind="grounding_multi", model_name="m", params=params
    )
    return scorers.get("grounding_multi").score(pd.DataFrame(rows), ctx)


def _row(answer, prediction, index="1"):
    return {"index": index, "answer": answer, "prediction": prediction}


GT = '{"bbox_2d": [100, 100, 200, 200], "label": "卡车"}'


def test_a_close_box_passes_and_reports_hit():
    out = _single([_row(GT, '{"bbox_2d": [105, 102, 203, 198]}')])
    assert out.loc[0, "pass_dev"] == 1
    assert out.loc[0, "hit"] == 1
    assert out.loc[0, "fail_bucket"] == ""
    assert out.loc[0, "dev_mean4_pct"] == pytest.approx(0.3)


def test_the_iou_gate_catches_a_box_whose_single_bad_point_is_averaged_away():
    """pred=[100,100,400,200] 对 gt=[100,100,200,200]：右边界偏了 200，另外三点全对，
    四点平均偏差正好 50 = 5%，光看平均会判「达标」，但这个框宽了两倍（IoU=0.33）。
    平均口径前面那道 IoU 门就是为这种情形加的。"""
    out = _single([_row(GT, '{"bbox_2d": [100, 100, 400, 200]}')])
    assert out.loc[0, "iou"] == pytest.approx(1 / 3)
    assert out.loc[0, "localized"] == 0
    assert out.loc[0, "pass_dev"] == 0
    assert out.loc[0, "fail_bucket"] == "localize_fail"


def test_deviation_columns_are_blank_when_localization_failed():
    """模型框到隔壁一辆车时四点偏差可能是 300，算进平均值会把整体拉成噪声。
    偏差只在 IoU 达门的样本上算。"""
    out = _single([_row(GT, '{"bbox_2d": [500, 500, 600, 600]}')])
    assert math.isnan(out.loc[0, "dev_mean4"])
    assert math.isnan(out.loc[0, "bias_x1"])
    assert out.loc[0, "localized"] == 0


def test_localization_rate_and_deviation_are_two_separate_numbers():
    """只报「成功样本上的偏差」会漂亮得离谱，只报「定位成功率」又丢了精度信息。"""
    out = _single(
        [
            _row(GT, '{"bbox_2d": [105, 102, 203, 198]}', "1"),
            _row(GT, '{"bbox_2d": [500, 500, 600, 600]}', "2"),
        ]
    )
    assert out["localized"].mean() == 0.5
    assert out["dev_mean4"].dropna().mean() == pytest.approx(3.0)


def test_malformed_output_counts_against_the_full_denominator():
    """空输出、非 JSON、拒答一律记不达标。只在解析成功的子集上算达标率，
    模型可以靠「拿不准就不输出」把分数刷上去。"""
    out = _single(
        [
            _row(GT, '{"bbox_2d": [105, 102, 203, 198]}', "1"),
            _row(GT, "图中没有找到符合描述的目标。", "2"),
            _row(GT, "", "3"),
        ]
    )
    assert out["hit"].tolist() == [1, 0, 0]
    assert out["hit"].mean() == pytest.approx(1 / 3)
    assert out["fail_bucket"].tolist() == ["", "malformed", "malformed"]


BIG_GT = '{"bbox_2d": [100, 100, 600, 600]}'


def test_the_three_failure_buckets_add_up_to_one_minus_the_pass_rate():
    out = _single(
        [
            _row(GT, '{"bbox_2d": [105, 102, 203, 198]}', "1"),      # 达标
            _row(GT, "拒答", "2"),                                     # 格式不合规
            _row(GT, '{"bbox_2d": [500, 500, 600, 600]}', "3"),      # 定位失败
            _row(BIG_GT, '{"bbox_2d": [100, 100, 600, 900]}', "4"),  # 框对了但不够准
        ]
    )
    assert out["fail_bucket"].tolist() == ["", "malformed", "localize_fail", "deviation"]
    pass_rate = out["hit"].mean()
    failures = (out["fail_bucket"] != "").mean()
    assert pass_rate + failures == 1.0


def test_code_fence_is_not_penalised():
    """对比 base 时的陷阱：base 会自己加围栏，解析器不宽容会让它因格式而虚低。"""
    out = _single([_row(GT, '```json\n{"bbox_2d": [105, 102, 203, 198]}\n```')])
    assert out.loc[0, "hit"] == 1
    assert "code_fence" in out.loc[0, "parse_flags"]


def test_strict_max4_column_is_reported_next_to_the_mean_one():
    """主指标用平均，严格版一起报出来 —— 两种读法都摆在桌面上比事后解释便宜。"""
    out = _single([_row(GT, '{"bbox_2d": [100, 100, 200, 260]}')])
    assert out.loc[0, "dev_mean4_pct"] == pytest.approx(1.5)
    assert out.loc[0, "dev_max4_pct"] == pytest.approx(6.0)
    assert out.loc[0, "pass_dev"] == 1
    assert out.loc[0, "pass_dev_strict"] == 0


def test_extra_boxes_on_a_single_box_task_are_flagged():
    out = _single([_row(GT, '[{"bbox_2d":[105,102,203,198]},{"bbox_2d":[0,0,10,10]}]')])
    assert "extra_boxes" in out.loc[0, "parse_flags"]
    assert out.loc[0, "hit"] == 1


def test_unparseable_ground_truth_is_excluded_from_every_denominator():
    """真值坏了不是模型的锅，记 0 会把它算成模型答错。"""
    out = _single([_row("这里本该是坐标", '{"bbox_2d": [105, 102, 203, 198]}')])
    assert pd.isna(out.loc[0, "hit"])
    assert out.loc[0, "fail_bucket"] == "gt_unparseable"


def test_thresholds_come_from_params():
    row = [_row(BIG_GT, '{"bbox_2d": [120, 120, 620, 620]}')]   # 四点各偏 20 = 2%
    assert _single(row, dev_threshold_pct=5.0).loc[0, "hit"] == 1
    assert _single(row, dev_threshold_pct=1.0).loc[0, "hit"] == 0
    assert _single(row, iou_gate=0.9).loc[0, "hit"] == 0


def test_scale_is_read_from_the_data_not_hardcoded():
    """构建端把 bbox_scale 逐样本写进 metadata。写死 1000 的话换个坐标空间会静默算错。"""
    rows = [
        {"index": "1", "answer": '{"bbox_2d": [50, 50, 100, 100]}',
         "prediction": '{"bbox_2d": [52, 52, 102, 102]}', "meta.bbox_scale": 500},
    ]
    out = _single(rows)
    assert out.loc[0, "dev_mean4_pct"] == pytest.approx(0.4)


def test_mixed_coordinate_spaces_are_rejected_instead_of_averaged():
    rows = [
        {"index": "1", "answer": '{"bbox_2d":[1,1,2,2]}', "prediction": '{"bbox_2d":[1,1,2,2]}',
         "meta.bbox_scale": 1000},
        {"index": "2", "answer": '{"bbox_2d":[1,1,2,2]}', "prediction": '{"bbox_2d":[1,1,2,2]}',
         "meta.bbox_scale": 500},
    ]
    with pytest.raises(ValueError, match="坐标空间必须一致"):
        _single(rows)


def test_scoring_is_deterministic():
    rows = [_row(GT, '{"bbox_2d": [105, 102, 203, 198]}', str(i)) for i in range(20)]
    first = _single(rows)["hit"].tolist()
    for _ in range(20):
        assert _single(rows)["hit"].tolist() == first


# ---------------------------------------------------------------- B 组多框

GT2 = '[{"bbox_2d": [0, 0, 100, 100]}, {"bbox_2d": [200, 200, 300, 300]}]'


def test_multi_box_matching_reports_prf_and_counts():
    out = _multi([_row(GT2, '[{"bbox_2d":[205,205,305,305]},{"bbox_2d":[2,1,101,99]}]')])
    assert out.loc[0, "n_matched"] == 2
    assert out.loc[0, "precision"] == 1.0 and out.loc[0, "recall"] == 1.0
    assert out.loc[0, "count_correct"] == 1
    assert out.loc[0, "count_error"] == 0


def test_missed_boxes_count_against_the_pass_rate():
    """漏检的框直接记不达标 —— 否则模型少输出几个框反而能把达标率做高。"""
    out = _multi([_row(GT2, '[{"bbox_2d":[0,0,100,100]}]')])
    assert out.loc[0, "n_missed"] == 1
    assert out.loc[0, "hit"] == 0.5
    assert out.loc[0, "count_error"] == -1


def test_spurious_boxes_hurt_precision_and_the_count():
    out = _multi([_row(GT2, '[{"bbox_2d":[0,0,100,100]},{"bbox_2d":[200,200,300,300]},{"bbox_2d":[700,700,800,800]}]')])
    assert out.loc[0, "n_spurious"] == 1
    assert out.loc[0, "precision"] == pytest.approx(2 / 3)
    assert out.loc[0, "recall"] == 1.0
    assert out.loc[0, "count_error"] == 1
    assert out.loc[0, "hit"] == 1.0     # 两个真值框都配上且够准


def test_count_accuracy_and_count_error_are_two_different_questions():
    """准确率只说「多少张图数对了」，误差说「数错时错多少」。错 1 个是边界目标的
    判断，错 10 个是模型压根没在数。"""
    out = _multi(
        [
            _row(GT2, '[{"bbox_2d":[0,0,100,100]}]', "1"),
            _row(GT2, '[{"bbox_2d":[0,0,100,100]},{"bbox_2d":[200,200,300,300]}]', "2"),
        ]
    )
    assert out["count_correct"].mean() == 0.5
    assert out["count_abs_error"].mean() == 0.5


def test_multi_box_deviation_is_averaged_over_matched_pairs_only():
    out = _multi([_row(GT2, '[{"bbox_2d":[2,2,102,102]},{"bbox_2d":[900,900,950,950]}]')])
    assert out.loc[0, "n_matched"] == 1
    assert out.loc[0, "dev_mean4"] == pytest.approx(2.0)
    assert out.loc[0, "n_spurious"] == 1


def test_empty_multi_box_prediction_scores_zero_not_nan():
    out = _multi([_row(GT2, "图中没有卡车。")])
    assert out.loc[0, "hit"] == 0.0
    assert out.loc[0, "recall"] == 0.0
    assert out.loc[0, "fail_bucket"] == "malformed"


def test_the_iou_gate_binds_harder_than_the_5_percent_rule_on_small_targets():
    """量化一下两道判据谁说了算：均匀平移 d 要保住 IoU>=0.5，需要 d <= 0.1835 * 边长。
    换算过来，只有边长超过图幅 27% 的框，5% 那条规则才比 IoU 门更紧；本批数据里的
    目标（equiv_px<96，即边长 < 图幅 9.4%）几乎全部由 IoU 门说了算。

    小框上偏 2%（20 个单位）就已经掉出 IoU 0.5，而 5% 规则还认为它达标。"""
    small = _single([_row(GT, '{"bbox_2d": [120, 120, 220, 220]}')])       # 边长 100
    assert small.loc[0, "iou"] < 0.5
    assert small.loc[0, "fail_bucket"] == "localize_fail"

    big = _single([_row(BIG_GT, '{"bbox_2d": [120, 120, 620, 620]}')])     # 边长 500
    assert big.loc[0, "iou"] > 0.5
    assert big.loc[0, "hit"] == 1
