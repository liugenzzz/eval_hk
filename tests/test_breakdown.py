"""§17 报表拆分：标灰规则、三种空格子、失败桶、加权总分、配对区间。"""

import math

import pandas as pd
import pytest

from eval_tool.aggregate import bootstrap_paired_diff_ci, summarize_metric, summarize_paired_diff
from eval_tool.breakdown import (
    BY_DESIGN,
    INSUFFICIENT,
    NOT_IN_DATA,
    OK,
    TREND_ONLY,
    EmptyCells,
    default_dims,
    make_breakdown,
    make_failure_buckets,
    make_weighted_total,
    parse_dims,
)


def _rows(n, dataset="g", model="sft", hit=lambda i: 1, **extra):
    return [
        {"model": model, "dataset": dataset, "index": f"{dataset}{i}", "hit": hit(i),
         "task_type": "ground_appearance", **{k: v(i) if callable(v) else v for k, v in extra.items()}}
        for i in range(n)
    ]


def test_a_cell_with_enough_samples_gets_a_score_and_an_interval():
    data = pd.DataFrame(_rows(40, hit=lambda i: int(i % 2)))
    out = make_breakdown(data, parse_dims([{"key": "task_type", "from": "task_type"}]), bootstrap_n=50)
    row = out.iloc[0]
    assert row["status"] == OK
    assert row["score"] == 0.5
    assert row["ci_low"] <= 0.5 <= row["ci_high"]


def test_a_small_cell_shows_n_but_no_percentage():
    """n=3 的 100% 会被当成结论。显示 n，不显示百分比。"""
    data = pd.DataFrame(_rows(5))
    out = make_breakdown(data, parse_dims([{"key": "task_type", "from": "task_type"}]), bootstrap_n=50)
    row = out.iloc[0]
    assert row["status"] == INSUFFICIENT
    assert row["n"] == 5
    assert math.isnan(row["score"])


def test_group_level_dimensions_are_greyed_out_even_with_enough_samples():
    """难度档在 task 级每格 n 只有几十，最坏 95% CI 半宽 ±13.9pp，只看趋势。"""
    data = pd.DataFrame(_rows(60, **{"meta.difficulty": lambda i: "easy" if i % 2 else "hard"}))
    dims = parse_dims([{"key": "difficulty", "from": "meta.difficulty", "level": "group"}])
    out = make_breakdown(data, dims, bootstrap_n=50)
    assert set(out["status"]) == {TREND_ONLY}
    assert out["score"].notna().all()      # 数还是给的，只是不下结论


def test_the_three_kinds_of_empty_cell_are_labelled_differently():
    """三者在报表上都是空白。实现的人看到空格会当 bug 修，必须分开标注。"""
    data = pd.DataFrame(_rows(40))
    out = make_breakdown(
        data,
        parse_dims([{"key": "task_type", "from": "task_type"},
                    {"key": "difficulty", "from": "meta.difficulty"}]),
        bootstrap_n=50,
        empty_cells=EmptyCells(by_design=(("ground_part", "hard"),), not_in_data=("ground_unique",)),
    )
    statuses = dict(zip(out["value"], out["status"]))
    assert statuses["ground_part × hard"] == BY_DESIGN
    assert statuses["ground_unique"] == NOT_IN_DATA
    # 第三种是抽到了但 n 不够，与前两种是不同的 status
    assert INSUFFICIENT not in {BY_DESIGN, NOT_IN_DATA}


def test_dimensions_can_be_restricted_to_certain_kinds():
    """尺寸档只对画框的组有意义，给计数任务报一个尺寸档纯属噪声。"""
    data = pd.DataFrame(_rows(40, **{"meta.size_bucket": "small"}))
    dims = parse_dims([{"key": "size_bucket", "from": "meta.size_bucket",
                        "only_kinds": ["grounding_single"]}])
    assert make_breakdown(data, dims, kinds={"g": "counting"}, bootstrap_n=50).empty
    assert not make_breakdown(data, dims, kinds={"g": "grounding_single"}, bootstrap_n=50).empty


def test_count_bins_are_computed_from_the_raw_count_column():
    data = pd.DataFrame(_rows(90, **{"meta.count": lambda i: 1 if i < 30 else (3 if i < 60 else 9)}))
    dims = parse_dims([{"key": "count_bin", "from": "meta.count",
                        "bins": [[1, 1, "单例"], [2, 5, "少量"], [6, None, "密集"]]}])
    out = make_breakdown(data, dims, bootstrap_n=50)
    assert sorted(out["value"]) == ["单例", "密集", "少量"]


def test_label_dimension_reports_only_the_worst_n():
    data = pd.DataFrame(
        _rows(150, hit=lambda i: int(i % 30 != 0), **{"meta.label": lambda i: f"类别{i // 30}"})
    )
    dims = parse_dims([{"key": "label", "from": "meta.label", "worst_n": 2}])
    out = make_breakdown(data, dims, bootstrap_n=50)
    assert len(out) == 2


def test_failure_buckets_add_up_to_one_minus_the_pass_rate():
    data = pd.DataFrame(
        [{"model": "sft", "dataset": "g", "fail_bucket": bucket}
         for bucket in ["", "", "malformed", "localize_fail", "deviation"]]
    )
    out = make_failure_buckets(data).iloc[0]
    assert out["pass_rate"] == 0.4
    assert out["fail_malformed"] + out["fail_localize_fail"] + out["fail_deviation"] == pytest.approx(0.6)


def test_unparseable_ground_truth_is_kept_out_of_the_bucket_denominator():
    """真值坏了不是模型的锅，算进分母会把达标率压低。"""
    data = pd.DataFrame(
        [{"model": "sft", "dataset": "g", "fail_bucket": b}
         for b in ["", "", "gt_unparseable"]]
    )
    out = make_failure_buckets(data).iloc[0]
    assert out["n"] == 2
    assert out["pass_rate"] == 1.0
    assert out["gt_unparseable_n"] == 1


def test_judge_scored_datasets_stay_out_of_the_acceptance_total():
    """裁判打的分会抖，同一份预测重跑两遍数字就不一样，做验收不合适。"""
    data = pd.DataFrame(_rows(40, dataset="ground_box") + _rows(40, dataset="describe"))
    out = make_weighted_total(
        data, {"ground_box": 0.4, "describe": 0.4}, {"ground_box": "code", "describe": "judge"}
    ).iloc[0]
    assert out["datasets_in_total"] == "ground_box"
    assert out["excluded_judge_datasets"] == "describe"


def test_acceptance_total_is_weighted_by_dataset():
    data = pd.DataFrame(
        _rows(40, dataset="ground_box", hit=lambda i: 1)
        + _rows(40, dataset="region_identify", hit=lambda i: 0)
    )
    engines = {"ground_box": "code", "region_identify": "code"}
    out = make_weighted_total(data, {"ground_box": 0.75, "region_identify": 0.25}, engines).iloc[0]
    assert out["total_score"] == 0.75


def test_small_datasets_are_named_rather_than_silently_dropped_from_the_total():
    data = pd.DataFrame(_rows(40, dataset="ground_box") + _rows(5, dataset="tiny"))
    out = make_weighted_total(data, {}, {"ground_box": "code", "tiny": "code"}).iloc[0]
    assert "tiny(n=5)" in out["excluded_small_n"]


# ------------------------------------------------------------ 配对 bootstrap

def test_paired_bootstrap_is_tighter_than_the_unpaired_one():
    """两个模型跑在同一批样本上。非配对区间把「两批不同样本」的抽样波动也算进去了，
    而那部分波动在配对设计里根本不存在。"""
    challenger = [1] * 55 + [0] * 45
    baseline = [1] * 50 + [0] * 50
    diff, low, high = bootstrap_paired_diff_ci(challenger, baseline, n_bootstrap=500, seed=1)
    assert diff == pytest.approx(0.05, abs=0.001)
    assert high - low < 0.30


def test_paired_bootstrap_ignores_samples_both_models_got_the_same_way():
    """两个模型在同一条难题上一起答错，那条对差值的贡献是 0，不该给区间贡献宽度。"""
    same = [1] * 100
    diff, low, high = bootstrap_paired_diff_ci(same, same, n_bootstrap=200, seed=1)
    assert diff == 0.0 and low == 0.0 and high == 0.0


def test_paired_bootstrap_requires_aligned_columns():
    with pytest.raises(ValueError, match="等长"):
        bootstrap_paired_diff_ci([1, 0], [1], n_bootstrap=10)


def test_paired_summary_flags_whether_the_interval_excludes_zero():
    rows = []
    for i in range(100):
        rows.append({"model": "sft", "dataset": "g", "index": str(i), "hit": 1})
        rows.append({"model": "base", "dataset": "g", "index": str(i), "hit": int(i % 2)})
    out = summarize_paired_diff(pd.DataFrame(rows), "base", bootstrap_n=200).iloc[0]
    assert out["diff"] == pytest.approx(0.5, abs=0.01)
    assert out["significant"]
    assert out["n_paired"] == 100


def test_continuous_metrics_can_be_summarised_too():
    """四点偏差和 IoU 是连续值，只接受 0/1 的那条路把它们挡在外面了。"""
    data = pd.DataFrame([{"dataset": "g", "dev_mean4_pct": v} for v in [1.0, 2.0, 3.0, 4.0]])
    out = summarize_metric(data, ["dataset"], "dev_mean4_pct", bootstrap_n=100).iloc[0]
    assert out["score"] == 2.5
    assert out["n"] == 4


def test_default_dims_cover_every_axis_the_report_needs():
    keys = {dim.key for dim in default_dims()}
    assert keys == {"task_type", "difficulty", "size_bucket", "label", "describe_kind",
                    "count_bin", "counting", "upstream_form", "answer_format",
                    "polarity", "hard_negative"}


def test_a_cell_with_no_usable_values_is_dropped_rather_than_greyed():
    """关了 pointwise 时 describe 的 hit 全是 NA，counting=zero 那一路按设计也不进
    F 组准确率。这不是「样本不足」，是这个指标对那一格不适用 —— 照样出行只会在
    报表里刷几十条无信息的灰格。"""
    data = pd.DataFrame(
        [{"model": "sft", "dataset": "d", "index": str(i), "task_type": "ground_full",
          "hit": None} for i in range(40)]
    )
    out = make_breakdown(data, parse_dims([{"key": "task_type", "from": "task_type"}]),
                         bootstrap_n=50)
    assert out.empty


# --- 按领域分类拆报表（书籍/装备共用的 category 轴） -----------------------


def test_quality_score_is_broken_down_on_its_own_0_100_scale():
    """通过率和均分回答的不是同一个问题，两个都得能拆开看。"""
    rows = (
        _rows(30, dataset="book_vqa", hit=lambda i: 1, category="作战应用",
              quality_score=lambda i: 80)
        + _rows(30, dataset="book_vqa", hit=lambda i: 1, category="拱形基础",
                quality_score=lambda i: 95)
    )
    for i, row in enumerate(rows):
        row["index"] = f"r{i}"

    breakdown = make_breakdown(
        pd.DataFrame(rows),
        parse_dims([{"key": "category", "from": "category"}]),
        metrics=["hit", "quality_score"],
    )

    quality = breakdown[breakdown["metric"] == "quality_score"].set_index("value")
    assert quality.loc["作战应用", "score"] == 80.0
    assert quality.loc["拱形基础", "score"] == 95.0
    # 两类通过率一样，均分差 15 分 —— 只看 hit 会以为两类一样好
    hits = breakdown[breakdown["metric"] == "hit"]
    assert set(hits["score"]) == {1.0}


def test_a_dimension_can_lower_its_own_sample_gate():
    """七大类每类只有十几条时，30 行的默认门槛会把整张表标灰。"""
    rows = _rows(12, dataset="book_vqa", category="作战应用", quality_score=lambda i: 70)

    default_gate = make_breakdown(
        pd.DataFrame(rows), parse_dims([{"key": "category", "from": "category"}])
    )
    lowered = make_breakdown(
        pd.DataFrame(rows),
        parse_dims([{"key": "category", "from": "category", "min_n": 10}]),
    )

    assert set(default_gate["status"]) == {INSUFFICIENT}
    assert set(lowered["status"]) == {OK}
