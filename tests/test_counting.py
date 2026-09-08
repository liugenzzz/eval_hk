"""F 组的文本解析：计数与清单。解析失败必须能和「数错了」分开。"""

import pytest

from eval_tool.counting import cn_to_int, count_bin, parse_count, parse_inventory, parse_inventory_gold


@pytest.mark.parametrize(
    "text,expected",
    [
        ("图中一共有 3 辆卡车。", 3),
        ("3", 3),
        ("有三辆", 3),
        ("十辆", 10),
        ("十二辆", 12),
        ("两辆", 2),
        ("图中没有货车。", 0),
        ("未发现该类目标", 0),
        ("0 个", 0),
    ],
)
def test_parse_count(text, expected):
    assert parse_count(text) == expected


@pytest.mark.parametrize("text", ["", None, "不好说", "画面比较模糊"])
def test_unparseable_count_returns_none_rather_than_zero(text):
    """抠不出数就是抠不出，返回 0 会把「没答」算成「答了 0」，那是两种失败。"""
    assert parse_count(text) is None


def test_chinese_numerals():
    assert cn_to_int("九") == 9
    assert cn_to_int("二十") == 20
    assert cn_to_int("二十一") == 21
    assert cn_to_int("百") is None


def test_parse_inventory_with_measure_words_and_separators():
    inventory = parse_inventory("有3名人员、2辆卡车和1艘船。")
    assert inventory.ok
    assert inventory.items == {"人员": 3, "卡车": 2, "船": 1}


def test_parse_inventory_matches_the_builder_answer_shape():
    inventory = parse_inventory("1名人员、1辆公交车、3辆卡车、1辆摩托车、1辆面包车。")
    assert inventory.items == {"人员": 1, "公交车": 1, "卡车": 3, "摩托车": 1, "面包车": 1}


def test_measure_word_is_optional_and_not_checked_against_a_table():
    """模型把「辆」说成「台」不该算它数错。"""
    assert parse_inventory("3台卡车").items == {"卡车": 3}
    assert parse_inventory("3卡车").items == {"卡车": 3}


def test_repeated_label_is_summed():
    assert parse_inventory("2辆卡车、3辆卡车").items == {"卡车": 5}


def test_unparseable_inventory_is_flagged_not_silently_empty():
    """解析不出来的整条记格式不合规，不计入准确率 —— 否则解析器的脆弱会被算成模型的错。"""
    result = parse_inventory("这张图看起来很复杂")
    assert not result.ok
    assert result.items == {}


def test_explicitly_empty_inventory_is_a_valid_answer():
    result = parse_inventory("图中没有清晰可见的目标。")
    assert result.ok
    assert result.items == {}


def test_gold_inventory_comes_from_metadata_in_several_shapes():
    assert parse_inventory_gold(["人员x3", "卡车x2"]).items == {"人员": 3, "卡车": 2}
    assert parse_inventory_gold('["人员x3"]').items == {"人员": 3}
    assert parse_inventory_gold("人员x3、卡车x2").items == {"人员": 3, "卡车": 2}
    assert parse_inventory_gold({"人员": 3}).items == {"人员": 3}


def test_malformed_gold_inventory_is_not_silently_treated_as_empty():
    assert not parse_inventory_gold(["人员"]).ok
    assert not parse_inventory_gold(None).ok


def test_count_bins_follow_the_tallyqa_split():
    """准确率随真值数量急剧衰减，合成一个总分会让 n=1 那批把 n>=6 的失败盖住。"""
    assert count_bin(1) == "单例"
    assert count_bin(2) == count_bin(5) == "少量"
    assert count_bin(6) == count_bin(20) == "密集"


def test_zero_gold_belongs_to_no_count_bin():
    """真值为 0 的那一路走拒答表，塞进「单例」会让那一档混进不是在数数的样本。"""
    assert count_bin(0) == ""
