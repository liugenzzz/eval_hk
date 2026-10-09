import json

import pandas as pd

from eval_tool import scorers
from eval_tool.scorers.aircraft import parse_category, parse_presence, parse_region
from eval_tool.scorers.aircraft import score_category, score_grounding, score_presence, score_region
from eval_tool.scorers import ScoringContext


def context(kind):
    return ScoringContext(dataset_key="aircraft", kind=kind, model_name="model")


def test_presence_parser_handles_negation_and_uncertainty():
    assert parse_presence("没有异常") is False
    assert parse_presence("没发现啥问题，外观完好，检查合格。") is False
    assert parse_presence("该部件不正常，可见裂纹") is True
    assert parse_presence("本项检查不通过，可见腐蚀。") is True
    assert parse_presence("无法确认是否缺失") is None
    data = pd.DataFrame([
        {"meta.image_status": "normal", "prediction": "未见缺陷"},
        {"meta.image_status": "anomalous", "prediction": "可见紧固件缺失"},
        {"meta.image_status": "anomalous", "prediction": "无法确认"},
    ])
    scored = score_presence(data, context("aircraft_presence"))
    assert scored["hit"].tolist() == [1, 1, 0]
    assert scored["unanswered"].tolist() == [0, 0, 1]


def test_category_uses_authoritative_gold_and_checks_letter_text_conflict():
    question = "图中是什么？\nA. 裂纹\nB. 腐蚀锈蚀\nC. 凹坑\nD. 划伤"
    assert parse_category("B", question) == ("corrosion", "")
    assert parse_category("B. 裂纹", question) == (None, "letter_text_conflict")
    assert parse_category("裂纹和凹坑", question) == (None, "multiple_classes")
    data = pd.DataFrame([
        {"meta.target_type": "corrosion", "question": question, "prediction": "B"},
        {"meta.target_type": "corrosion", "question": question, "prediction": "A"},
    ])
    assert score_category(data, context("aircraft_category"))["hit"].tolist() == [1, 0]


def test_grounding_requires_both_correct_class_and_location():
    gold = json.dumps([{"bbox_2d": [100, 100, 200, 200], "label": "紧固件缺失"}], ensure_ascii=False)
    correct = json.dumps([{"bbox_2d": [102, 101, 202, 201], "label": "紧固件缺失"}], ensure_ascii=False)
    wrong_class = json.dumps([{"bbox_2d": [100, 100, 200, 200], "label": "腐蚀锈蚀"}], ensure_ascii=False)
    data = pd.DataFrame([
        {"answer": gold, "prediction": correct},
        {"answer": gold, "prediction": wrong_class},
        {"answer": "[]", "prediction": "[]"},
        {"answer": gold, "prediction": f"```json\n{correct}\n```"},
    ])
    scored = score_grounding(data, context("aircraft_grounding"))
    assert scored["hit"].tolist() == [1, 0, 1, 0]
    assert scored["n_matched"].tolist() == [1, 0, 0, 1]
    assert scored["format_ok"].tolist() == [1, 1, 1, 0]
    assert scored.loc[1, "n_missed"] == 1
    assert scored.loc[1, "n_spurious"] == 1


def test_region_accepts_equivalent_position_phrases():
    assert parse_region("画面中下部") == "lower_center"
    assert parse_region("位于下方中部") == "lower_center"
    assert parse_region("在画面中间偏下") == "lower_center"
    assert parse_region("位于左侧中间位置") == "center_left"
    data = pd.DataFrame([
        {"answer": "缺陷位于画面中下部。", "prediction": "在下方中部。"},
        {"answer": "缺陷位于左上。", "prediction": "在右下。"},
    ])
    assert score_region(data, context("aircraft_region"))["hit"].tolist() == [1, 0]


def test_aircraft_scorers_register_without_changing_detection_kind():
    for name in ("aircraft_presence", "aircraft_category", "aircraft_grounding", "aircraft_region"):
        assert scorers.get(name).engine == scorers.CODE
    assert scorers.get("grounding_single").score is not score_grounding
