"""检测框解析与几何指标。

真实模型输出里这些都会出现，一个都不能崩 —— 见 docs 口径书 §3.5。
"""
from __future__ import annotations

import pandas as pd
import pytest

from eval_tool import bbox
from eval_tool.score_grounding import _score_multi, _score_one, match_boxes


# ---------------------------------------------------------------- 解析退化情形
@pytest.mark.parametrize(
    "text,n_boxes,flag",
    [
        ("", 0, "empty"),
        (None, 0, "empty"),
        ("模型今天不想干活", 0, "unparseable"),
        ('{"picked": []}', 0, "unparseable"),
        ('{"bbox_2d": [10, 20, 110, 220]}', 1, None),
        # base 模型爱自己加围栏。只计数，不扣分 ——
        # 扣分的话测出来的是格式差异不是能力差异。
        ('```json\n{"bbox_2d": [10, 20, 110, 220]}\n```', 1, "code_fence"),
        ('{"bbox_2d": [110, 220, 10, 20]}', 1, "reversed"),
        ('{"bbox_2d": [-5, 20, 1200, 220]}', 1, "out_of_range"),
        ('{"bbox_2d": [50, 50, 50, 50]}', 0, "degenerate"),
        ('{"bbox_2d": ["a", "b", "c", "d"]}', 0, "unparseable"),
        ("这个目标在 [10, 20, 110, 220] 这里", 1, "regex_fallback"),
    ],
)
def test_parse_handles_every_degenerate_case(text, n_boxes, flag):
    out = bbox.parse(text)
    assert len(out.boxes) == n_boxes
    if flag:
        assert flag in out.flags, out.flags


def test_reversed_and_out_of_range_are_repaired_not_dropped():
    """顺序颠倒和越界都是可修的，修完还要能算指标 —— 直接丢掉等于白白判模型失败。"""
    out = bbox.parse('{"bbox_2d": [1200, 220, -5, 20]}')
    assert out.boxes == [(0.0, 20.0, 1000.0, 220.0)]
    assert out.flags["reversed"] == 1 and out.flags["out_of_range"] == 1


def test_a_single_box_is_not_collected_twice():
    """命中 bbox_2d 键之后不能再递归进同一个 list。
    解析器造出来的重复框会被当成模型的重复输出，误检数直接翻倍。"""
    assert len(bbox.parse('{"bbox_2d": [1, 2, 3, 4]}').boxes) == 1
    assert len(bbox.parse('{"boxes": [{"bbox_2d": [1, 2, 3, 4]}]}').boxes) == 1


def test_model_emitted_duplicates_are_kept():
    """模型自己把同一个目标输出两次是【它的】问题，得留着计入误检，不能替它去重。"""
    out = bbox.parse('[{"bbox_2d": [1, 2, 3, 4]}, {"bbox_2d": [1, 2, 3, 4]}]')
    assert len(out.boxes) == 2


# ---------------------------------------------------------------- 几何
def test_iou_and_signed_deviation():
    assert bbox.iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0
    assert bbox.iou((0, 0, 100, 100), (200, 200, 300, 300)) == 0.0
    # 【有符号偏差看得出方向】x1 正、x2 负 = 系统性框小。
    # 绝对偏差看不出方向，看不出方向就不知道该补什么数据。
    dev = bbox.deviations((110, 110, 190, 190), (100, 100, 200, 200))
    assert dev["dev_x1"] > 0 and dev["dev_x2"] < 0
    assert dev["mae_4pt"] == 10.0


# ---------------------------------------------------------------- 单框打分
def test_deviation_is_only_computed_on_located_samples():
    """模型框到隔壁一辆车，四点偏差可能是 300，算进平均值会把整体拉成噪声。
    IoU >= 0.5 才算偏差，低于的记「定位失败」单独计数 —— 两个数分开报。"""
    hit = _score_one('{"bbox_2d": [12, 18, 108, 225]}', '{"bbox_2d": [10, 20, 110, 220]}')
    assert hit["located"] == 1 and hit["mae_4pt"] == 2.75

    miss = _score_one('{"bbox_2d": [500, 500, 600, 600]}', '{"bbox_2d": [10, 20, 110, 220]}')
    assert miss["located"] == 0
    assert pd.isna(miss["mae_4pt"]), "定位失败的样本不该进偏差统计"
    assert miss["iou"] == 0.0, "但 IoU 要照常记录"


def test_threshold_pass_uses_the_worst_corner_not_the_average():
    """Acc@dev<=N 要求【四个点全部】不超过 N。三个点很准、一个点差 80，
    平均下来还不到 20，但那个框已经漏掉了目标的一整条边。"""
    r = _score_one('{"bbox_2d": [100, 100, 200, 280]}', '{"bbox_2d": [100, 100, 200, 200]}')
    assert r["located"] == 1
    assert r["mae_4pt"] == 20.0          # 平均只有 20
    assert r["dev_le_20"] == 0           # 但最差那个角差了 80，不该通过


def test_extra_boxes_are_counted_not_silently_dropped():
    """问一个框却给了三个：取最接近的评能力，多出来的单独计数 ——
    把「多输出」和「找错目标」分开，两者的修法不一样。"""
    r = _score_one('[{"bbox_2d":[10,20,110,220]},{"bbox_2d":[500,500,600,600]}]',
                   '{"bbox_2d": [10, 20, 110, 220]}')
    assert r["located"] == 1 and r["extra_boxes"] == 1


def test_unparseable_gold_is_excluded_from_metrics_not_scored_zero():
    """真值本身坏了是【数据】的问题。判模型 0 分会把数据缺陷记到模型头上。"""
    r = _score_one('{"bbox_2d": [10, 20, 110, 220]}', "真值坏了")
    assert r["gold_bad"] == 1
    assert pd.isna(r["hit"]) and pd.isna(r["located"])


# ---------------------------------------------------------------- 多框打分
def test_matching_is_one_to_one_gated_and_optimal():
    """匹配要满足三条，用暴力枚举全部排列作对照验最后一条。

    用匈牙利而不是贪心：贪心按 IoU 降序配对，某些排布下先配掉的那对会挡住
    另外两对，整体少配上。模型输出没有置信度，无法像 COCO 那样按分数排序，
    匈牙利在这里是确定且最优的。
    """
    import itertools

    from eval_tool import bbox

    gold = [(0, 0, 100, 100), (30, 0, 130, 100), (500, 500, 600, 600)]
    pred = [(15, 0, 115, 100), (0, 0, 90, 100), (505, 495, 600, 600)]
    pairs = match_boxes(pred, gold)

    # 一对一：谁都不能被配两次
    assert len({i for i, _, _ in pairs}) == len(pairs)
    assert len({j for _, j, _ in pairs}) == len(pairs)
    # 只保留过阈值的
    assert all(v >= 0.5 for _, _, v in pairs)

    # 最优：总 IoU 不低于任何一种排列
    best = max(
        sum(bbox.iou(pred[i], gold[j]) for i, j in enumerate(perm)
            if bbox.iou(pred[i], gold[j]) >= 0.5)
        for perm in itertools.permutations(range(len(gold)))
    )
    assert sum(v for _, _, v in pairs) >= best - 1e-9


def test_matching_handles_empty_sides():
    """模型一个框都没给、或真值为空 —— 不能崩，返回空匹配。"""
    assert match_boxes([], [(0, 0, 10, 10)]) == []
    assert match_boxes([(0, 0, 10, 10)], []) == []


def test_count_accuracy_and_mae_are_reported_separately():
    """准确率只说「多少张图数对了」，MAE 说「数错时错多少」。
    错 1 个是边界目标的判断，错 10 个是模型压根没在数。"""
    gold = '[{"bbox_2d":[0,0,100,100]},{"bbox_2d":[200,200,300,300]},{"bbox_2d":[400,400,500,500]}]'
    r = _score_multi('[{"bbox_2d":[0,0,100,100]}]', gold)
    assert r["count_ok"] == 0 and r["count_mae"] == 2
    assert r["tp"] == 1 and r["fn"] == 2 and r["fp"] == 0
    assert r["recall"] == pytest.approx(1 / 3) and r["precision"] == 1.0

    exact = _score_multi(gold, gold)
    assert exact["count_ok"] == 1 and exact["count_mae"] == 0
    assert exact["f1"] == 1.0


def test_multi_with_no_match_reports_zero_f1_but_na_deviation():
    """一对都没配上时 F1 是 0（这是真实的能力评价），但偏差没有可算的对象，
    必须是 NA —— 填 0 会被当成「偏差为零」，把最差的情况读成最好的。"""
    r = _score_multi('[{"bbox_2d":[900,900,950,950]}]', '[{"bbox_2d":[0,0,100,100]}]')
    assert r["f1"] == 0.0
    assert pd.isna(r["mae_4pt"])


# ---------------------------------------------------------------- 硬约束
def test_scorers_are_deterministic():
    """engine=code 的打分器：同一份预测重跑一百遍，数字必须逐位相同。"""
    args = ('[{"bbox_2d":[12,18,108,225]},{"bbox_2d":[500,500,600,620]}]',
            '[{"bbox_2d":[10,20,110,220]},{"bbox_2d":[505,495,600,600]}]')
    first = _score_multi(*args)
    for _ in range(100):
        assert _score_multi(*args) == first


def test_scorers_never_touch_the_network():
    """engine=code 不许调任何模型 —— 一行网络请求都不能有。"""
    import socket

    def boom(*a, **k):
        raise AssertionError("打分器发起了网络请求")

    real_socket, real_conn = socket.socket, socket.create_connection
    socket.socket, socket.create_connection = boom, boom
    try:
        _score_one('{"bbox_2d":[1,2,3,4]}', '{"bbox_2d":[1,2,3,4]}')
        _score_multi('{"bbox_2d":[1,2,3,4]}', '{"bbox_2d":[1,2,3,4]}')
    finally:
        socket.socket, socket.create_connection = real_socket, real_conn
