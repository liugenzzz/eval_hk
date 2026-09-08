import pytest

from eval_tool.bbox import Box
from eval_tool.matching import match_boxes, solve_min_cost


def test_solve_min_cost_finds_the_global_optimum():
    # 贪心会先选 0（行 1 列 1），总代价 4+0+2=6；最优是 1+2+2=5。
    assert solve_min_cost([[4, 1, 3], [2, 0, 5], [3, 2, 2]]) == [1, 0, 2]


def test_solve_min_cost_handles_more_columns_than_rows():
    assignment = solve_min_cost([[9, 1, 9, 9]])
    assert assignment == [1]


def test_solve_min_cost_handles_more_rows_than_columns():
    assignment = solve_min_cost([[5], [1], [9]])
    assert assignment.count(-1) == 2
    assert assignment[1] == 0


@pytest.mark.parametrize("cost", [[], [[]]])
def test_solve_min_cost_on_empty_input(cost):
    assert all(v == -1 for v in solve_min_cost(cost))


def test_matching_is_order_independent():
    gt = [Box(0, 0, 100, 100), Box(200, 200, 300, 300)]
    pred = [Box(205, 205, 305, 305), Box(5, 5, 105, 105)]
    matching = match_boxes(gt, pred)
    assert matching.pairs == ((0, 1), (1, 0))
    assert matching.missed == () and matching.spurious == ()


def test_global_optimum_beats_greedy_on_two_adjacent_targets():
    """贪心按 IoU 从高到低抢配对时，挨得近的两个目标会被同一个预测框先抢走一个，
    于是同时多记一次漏检和一次误检 —— 看起来是模型漏了，其实是匹配挑错了。"""
    gt = [Box(0, 0, 100, 100), Box(50, 0, 150, 100)]
    pred = [Box(48, 0, 148, 100), Box(2, 0, 102, 100)]
    matching = match_boxes(gt, pred)
    assert matching.pairs == ((0, 1), (1, 0))
    assert not matching.missed and not matching.spurious


def test_pairs_below_the_gate_become_missed_and_spurious():
    matching = match_boxes([Box(0, 0, 100, 100)], [Box(500, 500, 600, 600)])
    assert matching.pairs == ()
    assert matching.missed == (0,)
    assert matching.spurious == (0,)


def test_no_predictions_is_all_missed():
    matching = match_boxes([Box(0, 0, 10, 10), Box(20, 20, 30, 30)], [])
    assert matching.missed == (0, 1)
    assert matching.spurious == ()


def test_no_ground_truth_is_all_spurious():
    matching = match_boxes([], [Box(0, 0, 10, 10)])
    assert matching.spurious == (0,)


def test_duplicate_prediction_counts_as_one_match_and_one_false_positive():
    """同一个目标输出两次：一个配上，另一个必须记成误检。"""
    matching = match_boxes([Box(0, 0, 100, 100)], [Box(0, 0, 100, 100), Box(0, 0, 100, 100)])
    assert len(matching.pairs) == 1
    assert len(matching.spurious) == 1


def test_matching_is_deterministic():
    gt = [Box(0, 0, 100, 100), Box(50, 50, 150, 150), Box(300, 300, 400, 400)]
    pred = [Box(52, 48, 152, 148), Box(298, 305, 398, 405), Box(1, 1, 99, 99)]
    first = match_boxes(gt, pred)
    assert all(match_boxes(gt, pred) == first for _ in range(50))
