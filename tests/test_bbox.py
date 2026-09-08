"""§3.5 列的解析退化情形。真实模型输出里这些都会出现，一个都不能崩。"""

import math

import pandas as pd
import pytest

from eval_tool.bbox import (
    BAD_BOX,
    CODE_FENCE,
    DUPLICATE,
    EMPTY,
    JSON_REPAIRED,
    OUT_OF_RANGE,
    REORDERED,
    UNPARSEABLE,
    Box,
    deviation,
    deviation_vs_object,
    iou,
    parse_boxes,
)


def test_plain_single_box():
    result = parse_boxes('{"bbox_2d": [10, 20, 30, 40], "label": "卡车"}')
    assert result.boxes == (Box(10, 20, 30, 40, "卡车"),)
    assert result.flags == frozenset()
    assert result.ok


def test_multi_box_array():
    result = parse_boxes('[{"bbox_2d":[1,2,3,4],"label":"a"},{"bbox_2d":[5,6,7,8],"label":"b"}]')
    assert len(result.boxes) == 2
    assert result.boxes[1].label == "b"


def test_bare_coordinate_array():
    assert parse_boxes("[10, 20, 30, 40]").boxes == (Box(10, 20, 30, 40),)


@pytest.mark.parametrize("text", ["", "   ", None, float("nan")])
def test_empty_output(text):
    result = parse_boxes(text)
    assert not result.ok
    assert EMPTY in result.flags
    assert UNPARSEABLE in result.flags


def test_non_json_text_yields_no_boxes_without_raising():
    result = parse_boxes("图中没有找到符合描述的目标。")
    assert not result.ok
    assert UNPARSEABLE in result.flags


def test_code_fence_is_stripped_and_only_counted():
    """Qwen3-VL base 会自己加围栏。解析器不宽容的话 base 的分会因【格式】而虚低，
    那测的是格式差异不是能力差异。围栏只计数，不影响坐标。"""
    result = parse_boxes('```json\n{"bbox_2d": [10, 20, 30, 40]}\n```')
    assert result.boxes == (Box(10, 20, 30, 40),)
    assert CODE_FENCE in result.flags


def test_reversed_coordinates_are_normalized_and_counted():
    result = parse_boxes('{"bbox_2d": [30, 40, 10, 20]}')
    assert result.boxes == (Box(10, 20, 30, 40),)
    assert REORDERED in result.flags


def test_out_of_range_is_clipped_and_counted():
    result = parse_boxes('{"bbox_2d": [-5, 20, 1200, 40]}')
    assert result.boxes == (Box(0, 20, 1000, 40),)
    assert OUT_OF_RANGE in result.flags


def test_scale_itself_is_a_legal_coordinate():
    """构建端 pixel_to_bbox2d 会把贴边的框 clamp 到 scale，1000 是合法值不是越界。"""
    result = parse_boxes('{"bbox_2d": [0, 0, 1000, 1000]}')
    assert OUT_OF_RANGE not in result.flags


def test_truncated_json_is_rescued_by_regex_and_flagged():
    result = parse_boxes('{"bbox_2d": [10, 20, 30, 40], "lab')
    assert result.boxes == (Box(10, 20, 30, 40),)
    assert JSON_REPAIRED in result.flags


def test_box_with_wrong_arity_is_dropped_and_flagged():
    result = parse_boxes('[{"bbox_2d":[1,2,3]},{"bbox_2d":[5,6,7,8]}]')
    assert result.boxes == (Box(5, 6, 7, 8),)
    assert BAD_BOX in result.flags


def test_non_numeric_coordinate_is_dropped_and_flagged():
    result = parse_boxes('{"bbox_2d": ["a", 2, 3, 4]}')
    assert not result.ok
    assert BAD_BOX in result.flags


def test_duplicate_boxes_are_flagged_but_kept():
    """悄悄去重就等于替模型擦掉一次失败，多框任务里那一个应当被记成误检。"""
    result = parse_boxes('[{"bbox_2d":[1,2,3,4]},{"bbox_2d":[1,2,3,4]}]')
    assert len(result.boxes) == 2
    assert DUPLICATE in result.flags


def test_nested_json_object_is_searched():
    result = parse_boxes('{"result": {"objects": [{"bbox_2d": [1, 2, 3, 4]}]}}')
    assert result.boxes == (Box(1, 2, 3, 4),)


def test_alternate_bbox_key_is_accepted():
    assert parse_boxes('{"bbox": [1, 2, 3, 4]}').boxes == (Box(1, 2, 3, 4),)


def test_iou_basics():
    assert iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)) == 1.0
    assert iou(Box(0, 0, 10, 10), Box(20, 20, 30, 30)) == 0.0
    assert iou(Box(0, 0, 10, 10), Box(5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_iou_of_degenerate_zero_area_boxes_does_not_divide_by_zero():
    assert iou(Box(5, 5, 5, 5), Box(5, 5, 5, 5)) == 1.0
    assert iou(Box(5, 5, 5, 5), Box(1, 1, 1, 1)) == 0.0


def test_signed_bias_tells_a_shrunken_box_from_a_shifted_one():
    """bias_x1 > 0 且 bias_x2 < 0 = 系统性框小；四点同号 = 整体偏移。
    绝对偏差看不出方向，看不出方向就不知道该补什么数据。"""
    shrunk = deviation(Box(110, 110, 190, 190), Box(100, 100, 200, 200))
    assert shrunk.signed[0] > 0 and shrunk.signed[2] < 0
    shifted = deviation(Box(110, 110, 210, 210), Box(100, 100, 200, 200))
    assert all(v > 0 for v in shifted.signed)
    assert shrunk.abs4 == shifted.abs4 == (10, 10, 10, 10)


def test_mean_and_max_deviation_percentages_are_against_the_frame():
    dev = deviation(Box(0, 0, 100, 300), Box(0, 0, 100, 100), scale=1000)
    assert dev.signed == (0, 0, 0, 200)
    assert dev.mean4 == 50.0
    assert dev.mean4_pct == 5.0
    assert dev.max4_pct == 20.0


def test_object_relative_deviation_is_much_harsher_on_small_targets():
    """图幅相对的 5% 在小目标上比目标本身还大 —— 只看那把尺子会把小目标的失败盖掉。"""
    big = deviation_vs_object(Box(0, 0, 520, 520), Box(0, 0, 500, 500))
    small = deviation_vs_object(Box(0, 0, 40, 40), Box(0, 0, 20, 20))
    assert big < small
    assert math.isnan(deviation_vs_object(Box(0, 0, 10, 10), Box(5, 5, 5, 5)))


def test_parsing_is_deterministic_across_repeated_runs():
    """engine="code" 的打分器：同一份输入重跑一百遍必须逐位相同。"""
    text = '```json\n[{"bbox_2d":[30,40,10,20],"label":"x"},{"bbox_2d":[-1,0,1200,5]}]\n```'
    first = parse_boxes(text)
    assert all(parse_boxes(text) == first for _ in range(100))
