"""Scorers for the task types aircraft_eval_v2 adds."""

import pandas as pd
import pytest

from eval_tool import scorers
from eval_tool.scorers import ScoringContext
from eval_tool.scorers.aircraft import (parse_presence, score_counting, score_region_presence,
                                        score_uncertainty)


def ctx(kind):
    return ScoringContext(dataset_key="aircraft", kind=kind, model_name="m")


@pytest.mark.parametrize("text, expected", [
    # pair_compare speaks in differences from the reference part
    ("两张图对比下来没啥区别，这块电路板的状态跟参考图是一致的。", False),
    ("对照参考图仔细核对了待检部件，各处状态均一致。", False),
    ("待检件与参考件不一致：右上角有一处划伤。", True),
    ("存在明显差异，待检件边缘开裂。", True),
])
def test_pair_compare_phrasings(text, expected):
    assert parse_presence(text) is expected


def test_referring_region_gold_is_about_the_box_not_the_image():
    data = pd.DataFrame({
        "answer": ["检查 [89, 226, 143, 285]，未见异常。", "该区域可见一处腐蚀锈蚀。"],
        "prediction": ["该处有裂纹。", "有腐蚀。"],
    })
    scored = score_region_presence(data, ctx("aircraft_region_presence"))
    assert scored["gold_present"].tolist() == [0, 1]
    assert scored["hit"].tolist() == [0, 1]
    assert scored["false_positive"].tolist() == [1, 0]


def test_unreadable_referring_gold_is_an_error():
    with pytest.raises(ValueError):
        score_region_presence(pd.DataFrame({"answer": ["看图"], "prediction": ["有"]}), ctx("x"))


def test_counting_reads_the_stated_count_not_a_coordinate():
    gold = '共 2 处腐蚀锈蚀。\n[{"bbox_2d": [307, 540, 438, 714], "label": "腐蚀锈蚀"}, {"bbox_2d": [1, 2, 3, 4], "label": "腐蚀锈蚀"}]'
    data = pd.DataFrame({
        "answer": [gold] * 4,
        "prediction": [
            "共两处腐蚀。",                                               # Chinese numeral
            '```json\n[{"bbox_2d": [307, 540, 438, 714]}, {"bbox_2d": [5, 6, 7, 8]}]\n```',  # boxes only
            "共 3 处。\n[]",                                             # stated count wins
            "看不出来。",
        ],
    })
    scored = score_counting(data, ctx("aircraft_counting"))
    assert scored["gold_count"].tolist() == [2, 2, 2, 2]
    assert scored["pred_count"].tolist()[:3] == [2, 2, 3]
    assert scored["hit"].tolist() == [1, 1, 0, 0]


def test_uncertainty_credits_declining_only():
    data = pd.DataFrame({"answer": ["看不清"] * 3, "prediction": [
        "照片模糊，没法下结论，建议补拍。", "存在腐蚀。", "未见异常。"]})
    assert score_uncertainty(data, ctx("aircraft_uncertainty"))["hit"].tolist() == [1, 0, 0]


def test_new_kinds_are_code_scored():
    for kind in ("aircraft_region_presence", "aircraft_counting", "aircraft_uncertainty"):
        assert scorers.get(kind).engine == scorers.CODE
