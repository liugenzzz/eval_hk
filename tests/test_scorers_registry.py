"""打分器注册表。重构前是 run_eval 里写死的三元组 + if/else 分支。"""
from __future__ import annotations

import pandas as pd
import pytest

from eval_tool import scorers


def test_default_kind_reproduces_the_old_hardcoded_branch():
    """重构前是 `if dataset_key in {"mcq", "judge"}: 选项抽取 else: 裁判`。
    未声明 kind 的配置必须走出和以前一字不差的结果，否则老配置会静默变行为。"""
    assert scorers.default_kind("mcq") == scorers.CHOICE
    assert scorers.default_kind("judge") == scorers.CHOICE
    assert scorers.default_kind("vqa") == scorers.JUDGE_TEXT
    # 自定义键名重构前会落进 else 分支，现在也必须落进 judge_text
    assert scorers.default_kind("随便起的名字") == scorers.JUDGE_TEXT


def test_all_kinds_in_the_spec_are_registered():
    kinds = set(scorers.registered_kinds())
    assert {"choice", "judge_text", "grounding_single", "grounding_multi"} <= kinds


def test_unknown_kind_fails_loudly_with_the_available_list():
    """拼错 kind 的后果是跑到一半才炸，而那时推理已经花掉了。
    报错要直接告诉你有哪些可选，不用去翻源码。"""
    with pytest.raises(KeyError) as exc:
        scorers.get("groundingsingle")          # 少了下划线
    assert "grounding_single" in str(exc.value)


def test_duplicate_registration_is_rejected():
    """静默覆盖会让人对着一份根本没在跑的代码调半天。"""
    with pytest.raises(ValueError):
        scorers.register("choice")(lambda data, ctx: data)


def test_dispatch_goes_to_the_registered_scorer():
    data = pd.DataFrame([{"index": "1", "answer": '{"bbox_2d":[10,20,110,220]}',
                          "prediction": '{"bbox_2d":[12,18,108,225]}'}])
    ctx = scorers.ScoreContext(dataset_key="det", model_name="m")
    out = scorers.get("grounding_single")(data, ctx)
    assert out.loc[0, "located"] == 1 and out.loc[0, "mae_4pt"] == 2.75
