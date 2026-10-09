import pandas as pd
import pytest

from eval_tool import scorers
from eval_tool.circular import expand_circular, rotate
from eval_tool.scorers import ScoringContext
from eval_tool.scorers.choice import score_choice_circular

QUESTION = "<video>\n视频中的异常行为属于哪一类？\nA. 打架斗殴\nB. 人员摔倒\nC. 翻越围墙\nD. 人员聚集\n请只回答选项字母。"


def test_rotate_moves_the_answer_with_its_text():
    q, letter, text = rotate(QUESTION, "B", 1)
    assert "A. 人员聚集\nB. 打架斗殴\nC. 人员摔倒\nD. 翻越围墙" in q
    assert (letter, text) == ("C", "人员摔倒")
    assert rotate(QUESTION, "B. 人员摔倒", 3)[1:] == ("A", "人员摔倒")


def test_rotate_refuses_questions_it_cannot_read():
    with pytest.raises(ValueError):
        rotate("没有选项的问题", "A", 1)
    with pytest.raises(ValueError):
        rotate("A. 甲\nC. 丙", "A", 1)


def test_expand_circular_keeps_rotation_zero_as_the_original():
    record = {"id": "v1", "videos": ["a.mp4"],
              "conversations": [{"from": "human", "value": QUESTION}, {"from": "gpt", "value": "B"}],
              "metadata": {"behavior": "人员摔倒"}}
    out = expand_circular(record)
    assert [r["id"] for r in out] == ["v1__r0", "v1__r1", "v1__r2", "v1__r3"]
    assert out[0]["conversations"] == record["conversations"]
    assert [r["conversations"][1]["value"] for r in out] == ["B", "C", "D", "A"]
    assert all(r["metadata"]["circular_id"] == "v1" and r["metadata"]["circular_n"] == 4 for r in out)
    assert record["conversations"][1]["value"] == "B"          # the input is not mutated


def _frame(preds, answers=("B", "C", "D", "A"), group="v1", n=4):
    return pd.DataFrame({
        "index": [f"{group}__r{k}__t1" for k in range(len(preds))],
        "question": [QUESTION] * len(preds),
        "prediction": list(preds),
        "answer": list(answers[: len(preds)]),
        "meta.circular_id": [group] * len(preds),
        "meta.rotation": list(range(len(preds))),
        "meta.circular_n": [n] * len(preds),
    })


CTX = ScoringContext(dataset_key="park_behavior", kind="choice_circular", model_name="m")


def test_all_rotations_right_is_one_hit_counted_once():
    scored = score_choice_circular(_frame(["B", "C", "答案是D", "A"]), CTX)
    assert scored["hit"].tolist()[0] == 1
    assert scored["hit"].isna().tolist() == [False, True, True, True]
    assert scored["vanilla_hit"].tolist()[0] == 1


def test_one_wrong_rotation_fails_the_question_but_not_vanilla():
    scored = score_choice_circular(_frame(["B", "C", "B", "A"]), CTX)
    assert scored["hit"].tolist()[0] == 0
    assert scored["vanilla_hit"].tolist()[0] == 1


def test_ambiguous_answer_is_not_credited():
    scored = score_choice_circular(_frame(["B 或 C", "C", "D", "A"]), CTX)
    assert scored["extraction_method"].tolist()[0] == "ambiguous"
    assert scored["hit"].tolist()[0] == 0


def test_incomplete_group_draws_no_conclusion():
    scored = score_choice_circular(_frame(["B", "C", "D"]), CTX)
    assert pd.isna(scored["hit"].tolist()[0])
    assert scored["circular_complete"].tolist() == [False, False, False]


def test_registered_as_code_kind():
    assert scorers.get("choice_circular").engine == scorers.CODE
