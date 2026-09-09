"""C / E 组的同义兜底：代码判不了的那些行才问裁判。

§6：「答案不在类别表里（模型自创了词）才丢给裁判判一次是不是同义 —— 这是 C / E 组
唯一用到裁判的地方。」
"""

import pandas as pd
import pytest

from eval_tool import scorers
from eval_tool.synonym import SynonymVerdict, parse_synonym


class FakeJudge:
    class settings:  # noqa: N801
        fingerprint = "fp"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def judge_raw(self, prompt, user_text, image_b64=None):
        self.calls.append(user_text)
        return self.replies.pop(0) if self.replies else '{"same": false, "relation": "none"}'


@pytest.fixture
def classes_yaml(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text("names:\n  - 面包车\n  - 三轮车\n  - 卡车\n", encoding="utf-8")
    return str(path)


def _ident(rows, judge=None, **params):
    ctx = scorers.ScoringContext(dataset_key="reg", kind="object_ident", model_name="sft",
                                params=params, judge_client=judge, max_workers=1)
    return scorers.get("object_ident").score(pd.DataFrame(rows), ctx)


def _short(rows, judge=None, **params):
    ctx = scorers.ScoringContext(dataset_key="attr", kind="short_answer", model_name="sft",
                                 params=params, judge_client=judge, max_workers=1)
    return scorers.get("short_answer").score(pd.DataFrame(rows), ctx)


IDENT = [
    {"index": "1", "meta.label": "面包车", "answer": "是面包车。", "prediction": "这是小面包"},
    {"index": "2", "meta.label": "面包车", "answer": "是面包车。", "prediction": "这是面包车"},
]


def test_the_fallback_is_off_by_default(classes_yaml):
    """关着的时候这两组是纯代码打分器，重跑一百遍逐位相同。"""
    judge = FakeJudge([])
    out = _ident(IDENT, judge=judge, classes_yaml=classes_yaml)
    assert judge.calls == []
    assert out["hit"].tolist() == [0, 1]     # 自创词记 0，是个偏严的下界


def test_only_rows_the_code_could_not_decide_reach_the_judge(classes_yaml):
    """代码判得了的绝不问裁判：一是省钱，二是代码判的结果不会抖。"""
    judge = FakeJudge(['{"same": true, "relation": "exact", "reason": "俗称"}'])
    out = _ident(IDENT, judge=judge, classes_yaml=classes_yaml, judge_synonym=True)
    assert len(judge.calls) == 1             # 只问了「小面包」那一条
    assert "小面包" in judge.calls[0]
    assert out["hit"].tolist() == [1, 1]


def test_a_hypernym_verdict_is_counted_separately_not_promoted(classes_yaml):
    """裁判说模型答得更粗，那和代码判出来的上位命中是同一件事 —— 各自计数，
    不并进精确命中率，否则「答粗一点更安全」这种退化就被洗白了。"""
    judge = FakeJudge(['{"same": true, "relation": "hypernym", "reason": "更粗"}'])
    out = _ident([IDENT[0]], judge=judge, classes_yaml=classes_yaml, judge_synonym=True)
    assert out.loc[0, "hit"] == 0
    assert out.loc[0, "hit_hypernym"] == 1
    assert out.loc[0, "judge_relation"] == "hypernym"


def test_a_negative_verdict_leaves_the_row_wrong(classes_yaml):
    judge = FakeJudge(['{"same": false, "relation": "none", "reason": "两种车"}'])
    out = _ident([IDENT[0]], judge=judge, classes_yaml=classes_yaml, judge_synonym=True)
    assert out.loc[0, "hit"] == 0
    assert out.loc[0, "judge_same"] == 0


def test_a_judge_failure_records_na_not_a_negative_verdict(classes_yaml):
    """判失败和判「不是同义」是两件事。记 0 会在两个模型的自创词比例不同时
    把对比拉出方向性偏差。"""
    judge = FakeJudge(["不是 JSON", "还是不是"])
    out = _ident([IDENT[0]], judge=judge, classes_yaml=classes_yaml, judge_synonym=True)
    assert pd.isna(out.loc[0, "judge_same"])
    assert str(out.loc[0, "judge_reason"]).startswith("[judge_error]")
    assert out.loc[0, "hit"] == 0            # 兜底没成功，维持代码的判断


def test_short_answers_accept_any_synonym_relation():
    """「车身是白的」和「白色车身」意思一样、字不一样。短答案没有粒度关系可言。"""
    judge = FakeJudge(['{"same": true, "relation": "hyponym", "reason": "同义"}'])
    out = _short([{"index": "1", "meta.attribute": "白色车身", "answer": "白色车身。",
                   "prediction": "车身是白的"}], judge=judge, judge_synonym=True)
    assert out.loc[0, "hit"] == 1


def test_verdicts_are_cached_so_a_rerun_costs_nothing(classes_yaml, tmp_path):
    from eval_tool.cache import JsonlCache

    cache = JsonlCache(tmp_path / "c.jsonl", ("judge_fp", "model", "dataset", "index"))
    judge = FakeJudge(['{"same": true, "relation": "exact"}'])
    ctx = scorers.ScoringContext(
        dataset_key="reg", kind="object_ident", model_name="sft", max_workers=1,
        params={"classes_yaml": classes_yaml, "judge_synonym": True},
        judge_client=judge, judge_cache=cache,
    )
    scorers.get("object_ident").score(pd.DataFrame([IDENT[0]]), ctx)
    assert len(judge.calls) == 1
    scorers.get("object_ident").score(pd.DataFrame([IDENT[0]]), ctx)
    assert len(judge.calls) == 1             # 第二次全部命中缓存


# ------------------------------------------------------------- 判词解析

def test_parser_accepts_a_fence_and_fills_the_default_relation():
    assert parse_synonym('```json\n{"same": true}\n```') == SynonymVerdict(True, "exact", "")


@pytest.mark.parametrize("bad", [
    '{"same": false, "relation": "exact"}',      # 自相矛盾
    '{"same": true, "relation": "none"}',        # 自相矛盾
    '{"relation": "exact"}',                     # 缺 same
    '{"same": true, "relation": "怪东西"}',       # 非法取值
    "裁判忘了输出 JSON",
])
def test_parser_rejects_contradictory_or_malformed_verdicts(bad):
    with pytest.raises(ValueError):
        parse_synonym(bad)
