"""D 组打分器：代码前置检查 + 分维度裁判。"""

import math

import pandas as pd
import pytest

from eval_tool import scorers
from eval_tool.describe_rubric import DescribeVerdict, parse_describe


class FakeJudge:
    """只实现 judge_raw —— D 组走的就是这个通用入口，不碰装备那套 judge_pointwise。"""

    class settings:  # noqa: N801
        fingerprint = "fake-fp"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def judge_raw(self, system_prompt, user_text, image_b64=None):
        self.calls.append(user_text)
        return self.replies.pop(0) if self.replies else '{"correct":3,"grounded":3,"informative":3}'


def _score(rows, judge=None, do_pointwise=True, **params):
    ctx = scorers.ScoringContext(
        dataset_key="describe", kind="describe", model_name="sft", params=params,
        judge_client=judge, do_pointwise=do_pointwise, max_workers=1,
    )
    return scorers.get("describe").score(pd.DataFrame(rows), ctx)


def _row(prediction, kind="appearance", task="ground_appearance", index="1", **extra):
    return {"index": index, "task_type": task, "question": "这辆三轮车本身长什么样？",
            "answer": "深红色车身，支着遮阳篷。", "prediction": prediction,
            "meta.describe_kind": kind, "meta.label": "三轮车", **extra}


MUST_NOT = {"appearance": ["位于", "画面", "左侧"], "position": ["颜色", "车身"]}


def test_scope_violations_are_flagged_by_code(): 
    out = _score(
        [_row("深红色车身，位于画面左侧", index="1"),
         _row("深红色车身，支着一顶白色遮阳篷", index="2")],
        do_pointwise=False, must_not=MUST_NOT,
    )
    assert out["scope_ok"].tolist() == [0, 1]
    assert out.loc[0, "scope_violations"] == "位于,画面,左侧"


def test_scope_is_not_judged_when_no_word_list_is_configured():
    """没配词表就不出这个数，不能拿一个空词表冒充「全都合规」。"""
    out = _score([_row("深红色车身，位于画面左侧")], do_pointwise=False)
    assert pd.isna(out.loc[0, "scope_ok"])


def test_code_metrics_need_no_judge_call_at_all():
    """范围合规、CHAIR、空话率是 D 组最先该看的三个数，关了 pointwise 也要出。"""
    judge = FakeJudge([])
    out = _score([_row("深红色车身，位于画面左侧")], judge=judge, do_pointwise=False,
                 must_not=MUST_NOT)
    assert judge.calls == []
    assert out.loc[0, "scope_ok"] == 0
    assert out.loc[0, "is_filler"] == 0
    assert pd.isna(out.loc[0, "hit"])


def test_filler_answers_are_flagged():
    out = _score([_row("一辆三轮车", index="1"),
                  _row("深红色车身，支着一顶白色遮阳篷，车斗敞开着", index="2")],
                 do_pointwise=False)
    assert out["is_filler"].tolist() == [1, 0]


@pytest.fixture
def class_table(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text("names:\n  - 三轮车\n  - 人员\n  - 船\n  - 卡车\n", encoding="utf-8")
    return str(path)


def test_chair_uses_the_inventory_as_the_authoritative_class_set(class_table):
    rows = [
        _row("一辆三轮车旁边有一艘船", index="1",
             **{"meta.source_image": "a.jpg", "meta.inventory": ["三轮车x1", "人员x2"]}),
    ]
    out = _score(rows, do_pointwise=False, classes_yaml=class_table)
    assert out.loc[0, "chair_available"] == 1
    assert out.loc[0, "hallucinated_classes"] == "船"
    assert out.loc[0, "chair_s"] == 1


def test_chair_is_unavailable_without_an_inventory(class_table):
    """label 拼出来的集合是不完整的，用它算 CHAIR 会系统性高估幻觉。"""
    out = _score([_row("一辆三轮车旁边有一艘船")], do_pointwise=False, classes_yaml=class_table)
    assert out.loc[0, "chair_available"] == 0
    assert math.isnan(out.loc[0, "chair_i"])


def test_chair_is_unavailable_without_a_real_class_table():
    """兜底表只含评估集里出现过的 label。模型编出一个「船」，表里没有「船」，
    这个词根本不会被识别成类别，幻觉就被漏报了 —— 和集合不完整时的高估正好相反。"""
    rows = [
        _row("一辆三轮车旁边有一艘船", index="1",
             **{"meta.source_image": "a.jpg", "meta.inventory": ["三轮车x1"]}),
    ]
    out = _score(rows, do_pointwise=False)
    assert out.loc[0, "chair_available"] == 0


def test_upstream_form_splits_the_three_kinds_of_context():
    """文字指代 / 坐标回指 / 上文承接是三种不同的能力，平均成一个数就看不出
    是哪一种垮了。"""
    out = _score(
        [_row("x", task="ground_appearance", index="1"),
         _row("x", task="detect_describe", index="2"),
         _row("x", task="inventory_locate", index="3")],
        do_pointwise=False,
    )
    assert out["upstream_form"].tolist() == ["文字指代", "坐标回指", "上文承接"]


# ------------------------------------------------------------------ 裁判部分

def test_judge_scores_three_dimensions_separately():
    judge = FakeJudge(['{"rubric":"describe_v1","correct":5,"grounded":4,"informative":2,"reason":"ok"}'])
    out = _score([_row("深红色车身")], judge=judge, must_not=MUST_NOT)
    assert out.loc[0, "judge_correct"] == 5
    assert out.loc[0, "judge_grounded"] == 4
    assert out.loc[0, "judge_informative"] == 2
    assert out.loc[0, "hit"] == pytest.approx((5 + 4 + 2) / 15, abs=1e-4)


def test_a_missing_prediction_never_reaches_the_judge():
    judge = FakeJudge([])
    out = _score([_row("")], judge=judge)
    assert judge.calls == []
    assert pd.isna(out.loc[0, "hit"])
    assert out.loc[0, "judge_reason"] == "[missing_prediction]"


def test_a_judge_failure_records_none_not_zero():
    """判失败和判低分是两件事。记 0 会在两个模型失败率不同时把对比拉出方向性偏差。"""
    judge = FakeJudge(["这不是 JSON", "还是不是 JSON"])
    out = _score([_row("深红色车身")], judge=judge)
    assert pd.isna(out.loc[0, "hit"])
    assert out.loc[0, "judge_reason"].startswith("[judge_error]")


def test_the_judge_is_retried_once_before_giving_up():
    judge = FakeJudge(["坏的", '{"correct":4,"grounded":4,"informative":4}'])
    out = _score([_row("深红色车身")], judge=judge)
    assert len(judge.calls) == 2
    assert out.loc[0, "hit"] == pytest.approx(0.8)


# ------------------------------------------------------------------ rubric 解析

def test_rubric_parser_accepts_a_code_fence():
    verdict = parse_describe('```json\n{"correct":5,"grounded":5,"informative":5}\n```')
    assert verdict.mean_normalized == 1.0


def test_rubric_parser_rejects_a_missing_dimension():
    with pytest.raises(ValueError, match="缺少维度"):
        parse_describe('{"correct":4,"grounded":4}')


def test_rubric_parser_rejects_an_out_of_range_score():
    with pytest.raises(ValueError, match="超出"):
        parse_describe('{"correct":9,"grounded":1,"informative":1}')


def test_mean_is_only_a_handle_the_three_dimensions_stay_split():
    """「正确但空洞」和「具体但在编」平均下来可以是同一个数。"""
    a = DescribeVerdict(correct=5, grounded=5, informative=1, reason="")
    b = DescribeVerdict(correct=1, grounded=5, informative=5, reason="")
    assert a.mean_normalized == b.mean_normalized
    assert a.as_columns()["judge_correct"] != b.as_columns()["judge_correct"]
