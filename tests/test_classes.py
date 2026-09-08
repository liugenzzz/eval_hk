"""类别表与上下位判定。算法与构建端 core/classes.py::_detect_confusable 一致。"""

import pytest

from eval_tool.classes import (
    EXACT,
    HYPERNYM,
    HYPONYM,
    OFF_TABLE,
    OTHER,
    load_class_table,
    one_char_apart,
    table_from_names,
)

TABLE = table_from_names(
    ["人员", "军事人员", "三轮车", "遮阳三轮车", "卡车", "货车", "切管器", "切管机", "面包车"]
)


def test_exact_match():
    assert TABLE.relation("遮阳三轮车", "遮阳三轮车") == EXACT


def test_answering_coarser_is_a_hypernym_hit_not_a_plain_error():
    """金标「遮阳三轮车」答「三轮车」：答粗了。安全但是退化，必须单独计数。"""
    assert TABLE.relation("遮阳三轮车", "三轮车") == HYPERNYM


def test_answering_finer_is_a_hyponym_hit_and_is_not_the_same_failure():
    """金标「三轮车」答「遮阳三轮车」：答细了 —— 在幻觉一个它看不清的属性。
    构建端的 hypernym 组是对称的，不拆方向就会把这两种病混成一个数。"""
    assert TABLE.relation("三轮车", "遮阳三轮车") == HYPONYM


def test_unrelated_class_is_a_plain_error():
    assert TABLE.relation("三轮车", "卡车") == OTHER


def test_a_word_outside_the_table_is_off_table_not_wrong():
    """模型自创了词才丢给裁判判一次是不是同义 —— 这是 C/E 组唯一用到裁判的地方。"""
    assert TABLE.relation("三轮车", "小电驴") == OFF_TABLE
    assert TABLE.relation("三轮车", "") == OFF_TABLE


def test_longest_match_wins_when_extracting_from_a_sentence():
    """「遮阳三轮车」不能被抠成「三轮车」，否则模型答对了也会被判成上位命中。"""
    assert TABLE.find_in_text("该区域内的是遮阳三轮车。") == "遮阳三轮车"
    assert TABLE.find_in_text("这是三轮车") == "三轮车"
    assert TABLE.find_in_text("这是一架飞机") is None


def test_one_char_apart_classes_are_confusable_but_not_hypernyms():
    """切管器 vs 切管机是并列的不同东西，不是上下位。"""
    assert one_char_apart("切管器", "切管机")
    assert TABLE.relation("切管器", "切管机") == OTHER
    assert TABLE.is_confusable("切管器", "切管机")


def test_unrelated_classes_are_not_confusable():
    assert not TABLE.is_confusable("卡车", "人员")


def test_normalization_ignores_case_and_spaces():
    table = table_from_names(["SUV", "面包车"])
    assert table.relation("SUV", " suv ") == EXACT


def test_load_class_table_from_yaml_mapping(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text("nc: 2\nnames:\n  0: 人员\n  1: 卡车\n", encoding="utf-8")
    table = load_class_table(path)
    assert table.count == 2
    assert table.canonical("卡车") == "卡车"


def test_load_class_table_from_yaml_list(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text("names:\n  - 人员\n  - 卡车\n", encoding="utf-8")
    assert load_class_table(path).count == 2


def test_load_class_table_from_json(tmp_path):
    """内网机器上没装 PyYAML 时可以用等价的 JSON 表。"""
    path = tmp_path / "classes.json"
    path.write_text('{"names": ["人员", "卡车"]}', encoding="utf-8")
    assert load_class_table(path).count == 2


def test_declared_class_count_must_match(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text("nc: 5\nnames:\n  - 人员\n", encoding="utf-8")
    with pytest.raises(ValueError, match="nc=5"):
        load_class_table(path)


def test_missing_class_table_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_class_table(tmp_path / "nope.yaml")
