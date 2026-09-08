"""§10 格式合规 / 任务串味 / 截断。微调模型特有的失败模式。"""

import pandas as pd
import pytest

from eval_tool.compliance import BOXES, CHOICE, NUMBER, TEXT, attach_compliance, compliance_row, detect_form


def test_detect_form_tells_boxes_from_prose_from_numbers():
    assert detect_form('{"bbox_2d": [1, 2, 3, 4]}') == BOXES
    assert detect_form("3") == NUMBER
    assert detect_form("一辆深红色的三轮车，车斗敞开着。") == TEXT
    assert detect_form("") == "empty"


def test_code_fence_never_costs_format_compliance():
    """base 会自己加围栏。解析器不宽容会让它因格式而虚低 —— 那测的是格式差异
    不是能力差异。围栏只计数。"""
    record = compliance_row('```json\n{"bbox_2d": [1, 2, 3, 4]}\n```', BOXES)
    assert record["format_ok"] == 1
    assert record["has_code_fence"] == 1


def test_out_of_range_coordinates_are_counted_but_still_parse():
    record = compliance_row('{"bbox_2d": [-5, 0, 1200, 10]}', BOXES)
    assert record["format_ok"] == 1
    assert record["coord_out_of_range"] == 1


def test_prose_where_boxes_were_asked_is_task_bleed():
    """问框答文字。"""
    record = compliance_row("图中那辆三轮车停在路边。", BOXES)
    assert record["format_ok"] == 0
    assert record["task_bleed"] == 1
    assert record["bleed_kind"] == "text_for_boxes"


def test_coordinates_where_a_description_was_asked_is_task_bleed():
    """问描述吐坐标。"""
    record = compliance_row('{"bbox_2d": [1, 2, 3, 4]}', TEXT)
    assert record["task_bleed"] == 1
    assert record["bleed_kind"] == "boxes_for_text"


def test_empty_output_is_not_counted_as_bleed():
    """空输出是「没答」，不是「答错了形态」，两种失败要分开。"""
    record = compliance_row("", BOXES)
    assert record["format_ok"] == 0
    assert record["task_bleed"] == 0


def test_choice_format_requires_an_option_letter():
    assert compliance_row("答案是 B", CHOICE)["format_ok"] == 1
    assert compliance_row("我觉得都不对", CHOICE)["format_ok"] == 0


def test_number_format_requires_something_countable():
    assert compliance_row("图中有 3 辆卡车", NUMBER)["format_ok"] == 1
    assert compliance_row("不好说", NUMBER)["format_ok"] == 0


def test_truncated_json_is_detected_from_unbalanced_brackets():
    record = compliance_row('{"bbox_2d": [1, 2, 3, 4], "label": "卡', BOXES)
    assert record["truncation_suspected"] == 1
    assert record["truncation_evidence"] == "unbalanced_brackets"


def test_long_text_without_terminal_punctuation_is_only_suspected():
    """自由文本那一档只是嫌疑 —— 精确的「撞上 max_new_tokens」要推理侧给 token 数。"""
    long_text = "深红色车身，支着一顶白色遮阳篷，篷布一角有些破损，车斗敞开着，里面堆着几个纸箱"
    assert compliance_row(long_text, TEXT)["truncation_suspected"] == 1
    assert compliance_row(long_text + "。", TEXT)["truncation_suspected"] == 0


def test_short_text_is_not_flagged_as_truncated():
    assert compliance_row("白色车身", TEXT)["truncation_suspected"] == 0


def test_attach_compliance_does_not_overwrite_a_scorer_column():
    data = pd.DataFrame([{"prediction": "图中那辆车。", "format_ok": 1}])
    out = attach_compliance(data, BOXES)
    assert out.loc[0, "format_ok"] == 1      # 打分器自己算过的不被合规层覆盖
    assert out.loc[0, "task_bleed"] == 1


def test_attach_compliance_is_a_no_op_without_predictions():
    data = pd.DataFrame([{"index": "1"}])
    assert attach_compliance(data, BOXES).equals(data)
