"""D 组的代码前置检查：范围合规、CHAIR 幻觉、空话率。

这三个是 D 组唯一不受裁判偏置影响的客观锚 —— 裁判和被测是同家族，没有异家族裁判
可以做自偏检测，所以它们必须与裁判分并列报。
"""

import pytest

from eval_tool.classes import table_from_names
from eval_tool.describe_scope import (
    chair,
    is_filler,
    load_scope_rules,
    parse_scope_rule,
    rules_from_mapping,
)

TABLE = table_from_names(["人员", "三轮车", "遮阳三轮车", "卡车", "船", "面包车"])


def _write_prompt(directory, name, kind, must_not):
    (directory / f"{name}.txt").write_text(
        f"# 注释行\n#! kind: {kind}\n#! needs: 任意目标\n#! must-not: {must_not}\n"
        "answer-spec: 只写这个物体自己身上的东西\n",
        encoding="utf-8",
    )


def test_scope_rules_are_read_from_the_builder_prompt_files(tmp_path):
    """词表复用构建这批数据时的同一份约束。重写一遍必然对不上。"""
    _write_prompt(tmp_path, "appearance", "appearance", "位于 画面 方位 左侧")
    _write_prompt(tmp_path, "position", "position", "颜色 车身")
    rules = load_scope_rules(tmp_path)
    assert rules["appearance"].must_not == ("位于", "画面", "方位", "左侧")
    assert rules["position"].must_not == ("颜色", "车身")


def test_an_empty_must_not_line_yields_no_words(tmp_path):
    """contrast 和 full 本来就没有禁用词。正则如果用 \\s* 会把换行吃掉，
    把下一行整行抓成词表，凭空给这两个 kind 造出一堆禁用词。"""
    (tmp_path / "contrast.txt").write_text(
        "#! kind: contrast\n#! must-not:\nanswer-spec: 只写差异 —— 和另外几个同类比\n",
        encoding="utf-8",
    )
    rules = load_scope_rules(tmp_path)
    assert rules["contrast"].must_not == ()
    assert rules["contrast"].violations("和另外几个同类比，它的车身更长") == ()


def test_a_prompt_without_a_kind_tag_is_skipped(tmp_path):
    (tmp_path / "readme.txt").write_text("这不是一个描述提示词\n", encoding="utf-8")
    _write_prompt(tmp_path, "appearance", "appearance", "位于")
    assert set(load_scope_rules(tmp_path)) == {"appearance"}


def test_a_directory_with_no_describe_prompts_raises(tmp_path):
    (tmp_path / "readme.txt").write_text("nothing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="describe 提示词"):
        load_scope_rules(tmp_path)


def test_off_topic_answers_are_caught_by_code_not_by_the_judge():
    """通用的「描述准确性」rubric 会给跑题答案高分（说得没错啊），
    裁判判不出「跑题」这件事。"""
    rule = parse_scope_rule("#! kind: appearance\n#! must-not: 位于 画面 左侧\n")
    assert rule.violations("深红色车身，位于画面左侧") == ("位于", "画面", "左侧")
    assert rule.violations("深红色车身，支着一顶白色遮阳篷") == ()


def test_rules_can_come_from_config_when_the_builder_dir_is_unavailable():
    rules = rules_from_mapping({"appearance": ["位于", "画面"]})
    assert rules["appearance"].violations("位于画面右侧") == ("位于", "画面")


# --------------------------------------------------------------- CHAIR 幻觉

def test_chair_counts_classes_mentioned_but_not_present():
    result = chair("一辆三轮车旁边有一艘船", ["三轮车", "人员"], TABLE)
    assert result.mentioned == ("三轮车", "船")
    assert result.hallucinated == ("船",)
    assert result.chair_i == 0.5
    assert result.chair_s == 1


def test_a_clean_description_scores_zero_on_both_granularities():
    result = chair("一辆三轮车旁边站着一名人员", ["三轮车", "人员"], TABLE)
    assert result.chair_i == 0.0
    assert result.chair_s == 0


def test_chair_is_not_computed_without_an_authoritative_class_set():
    """用不完整的集合算 CHAIR 会系统性高估幻觉 —— 图里真实存在但没进评估集标注的
    目标，会被一个不落地记成模型编的。宁可标「无法计算」也不要给一个偏的数。"""
    result = chair("一辆三轮车", None, TABLE)
    assert not result.available
    assert result.chair_i is None and result.chair_s is None


def test_a_hyponym_is_not_double_counted():
    """「遮阳三轮车」里含「三轮车」。多数一个类别就是凭空多一次幻觉。"""
    result = chair("一辆遮阳三轮车", ["遮阳三轮车"], TABLE)
    assert result.mentioned == ("遮阳三轮车",)
    assert result.hallucinated == ()


# --------------------------------------------------------------- 空话率

def test_filler_answers_are_detected():
    """「一辆车」这种等于没说的话。信息量里能用代码抓的那一半。"""
    assert is_filler("一辆三轮车", "三轮车")
    assert is_filler("三轮车。", "三轮车")
    assert is_filler("")


def test_a_substantive_description_is_not_filler():
    assert not is_filler("深红色车身，支着一顶白色遮阳篷，车斗敞开着", "三轮车")
