"""Judge replies seen from Qwen3.8-27B on the detection run that used to error out."""

import json

import pytest

from eval_tool.classes import EXACT
from eval_tool.describe_rubric import parse_describe
from eval_tool.judge_rubrics import extract_json_object
from eval_tool.synonym import parse_synonym


def test_synonym_relation_counts_as_exact():
    verdict = parse_synonym('{"rubric": "synonym_v1", "same": true, "relation": "synonym", "reason": "同义"}')
    assert (verdict.same, verdict.relation) == (True, EXACT)


def test_truncated_describe_verdict_keeps_its_scores():
    reply = ('{"rubric": "describe_v1", "correct": 3, "grounded": 5, "informative": 5, '
             '"reason": "车牌号码识别错误（图中为P211 YGJ，回答为P211 YGJ，实际上看起来像P211 YGJ，等等，')
    verdict = parse_describe(reply)
    assert (verdict.correct, verdict.grounded, verdict.informative) == (3, 5, 5)


def test_truncated_verdict_missing_a_score_is_still_an_error():
    with pytest.raises(ValueError):
        parse_describe('{"rubric": "describe_v1", "correct": 3, "grounded": 5, "reason": "写到一半')


# Industrial verdicts from the industrial run that errored out: every score is there,
# but the free-text reason broke the JSON.
UNESCAPED_QUOTE = ('{\n  "rubric": "industrial_v1",\n  "reference_status": "usable",\n  "object_score": 100,\n'
                   '  "defect_score": 30,\n  "appearance_score": 40,\n  "location_score": 60,\n'
                   '  "fabrication_score": 60,\n  "style_score": 90,\n  "critical_errors": [\n'
                   '    "wrong_defect_type"\n  ],\n  "reasoning": "模型甚至明确否认了"chunk missing"，'
                   '与图片中 "fact_score": 0 式的描述无关。"\n}')
DOUBLED_QUOTE = ('{\n"rubric": "industrial_v1",\n"reference_status": "usable",\n"object_score": 100,\n'
                 '"defect_score": 100,\n"appearance_score": 90,\n"location_score": 80,\n'
                 '""fabrication_score": 80,\n"style_score": 100,\n"critical_errors": [],\n'
                 '"reasoning": "裂纹位置描述正确。"\n}')


def test_verdict_with_broken_reason_keeps_its_scores():
    obj = extract_json_object(UNESCAPED_QUOTE)
    assert obj["rubric"] == "industrial_v1"
    assert (obj["defect_score"], obj["fabrication_score"]) == (30, 60)
    assert obj["critical_errors"] == ["wrong_defect_type"]
    assert "fact_score" not in obj            # a field name quoted inside the reason is not read
    assert obj["reasoning"].startswith("模型甚至明确否认了")
    assert extract_json_object(DOUBLED_QUOTE)["fabrication_score"] == 80


def test_reply_without_a_verdict_still_errors():
    with pytest.raises(json.JSONDecodeError):
        extract_json_object('我认为回答基本正确，"但有一处" 错误')
