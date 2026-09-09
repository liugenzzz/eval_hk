"""CHAIR 的分母：每张图的完整类别集合。

三个来源按可信度排：metadata.image_classes（构建端将来给的）> 原始 YOLO 标注文件
（每条样本都带 source_annotation，覆盖 100%）> metadata.inventory（只有
inventory_locate 那一种样本带，覆盖不到四分之一）。
"""

import pandas as pd
import pytest

from eval_tool import scorers
from eval_tool.classes import load_class_table, table_from_names
from eval_tool.image_classes import ImageClassIndex, declared_image_classes


@pytest.fixture
def labels(tmp_path):
    d = tmp_path / "labels"
    d.mkdir()
    # YOLO: class_id cx cy w h
    (d / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n2 0.2 0.2 0.05 0.05\n0 0.8 0.8 0.1 0.1\n",
                             encoding="utf-8")
    (d / "empty.txt").write_text("", encoding="utf-8")
    (d / "bad_id.txt").write_text("99 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    (d / "garbage.txt").write_text("这不是标注\n", encoding="utf-8")
    return d


@pytest.fixture
def table():
    return table_from_names(["人员", "卡车", "三轮车"])


def test_class_set_comes_from_the_annotation_file(labels, table):
    index = ImageClassIndex(labels, table)
    assert index.classes_of("a.txt") == ("三轮车", "人员")   # class_id 0 和 2，去重排序


def test_an_image_with_no_boxes_yields_an_empty_set_not_none(labels, table):
    """空标注是「这张图什么都没有」，和「读不到」是两件事：前者下模型提任何类别
    都是幻觉，后者应当标成无法计算。"""
    assert ImageClassIndex(labels, table).classes_of("empty.txt") == ()


def test_a_class_id_missing_from_the_table_gives_up_instead_of_a_partial_set(labels, table):
    """标注和类别表对不上时，硬算出来的集合是残缺的 —— 拿它去算 CHAIR，
    图里真实存在的那一类会被记成模型编的。"""
    assert ImageClassIndex(labels, table).classes_of("bad_id.txt") is None


def test_unreadable_annotation_gives_up(labels, table):
    assert ImageClassIndex(labels, table).classes_of("garbage.txt") is None
    assert ImageClassIndex(labels, table).classes_of("nope.txt") is None
    assert ImageClassIndex(labels, table).classes_of("") is None


def test_the_index_is_disabled_without_a_labels_dir_or_table(labels, table):
    assert not ImageClassIndex(None, table).enabled
    assert not ImageClassIndex(labels, None).enabled
    assert ImageClassIndex(None, table).classes_of("a.txt") is None


def test_repeated_lookups_read_the_file_once(labels, table):
    index = ImageClassIndex(labels, table)
    first = index.classes_of("a.txt")
    (labels / "a.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    assert index.classes_of("a.txt") == first     # 同一张图上有十几条样本，读一次就够


def test_declared_image_classes_wins_when_the_builder_provides_it():
    assert declared_image_classes({"meta.image_classes": ["人员", "卡车"]}) == ("人员", "卡车")
    assert declared_image_classes({"meta.image_classes": "人员,卡车"}) == ("人员", "卡车")
    assert declared_image_classes({"meta.image_classes": []}) is None
    assert declared_image_classes({}) is None


# ------------------------------------------------------- 接进 D 组打分器

def _describe_rows(**extra):
    return pd.DataFrame([{
        "index": "1", "task_type": "ground_appearance", "question": "长什么样？",
        "answer": "一个深灰色的目标。", "prediction": "一辆卡车旁边还有一辆三轮车。",
        "meta.label": "卡车", "meta.source_image": "a.jpg", "meta.source_annotation": "a.txt",
        **extra,
    }])


def _score(rows, classes_yaml, **params):
    ctx = scorers.ScoringContext(
        dataset_key="describe", kind="describe", model_name="sft", do_pointwise=False,
        params={"classes_yaml": classes_yaml, **params},
    )
    return scorers.get("describe").score(rows, ctx)


@pytest.fixture
def classes_yaml(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text("names:\n  - 人员\n  - 卡车\n  - 三轮车\n", encoding="utf-8")
    return str(path)


def test_chair_is_computable_for_every_row_once_labels_dir_is_configured(labels, classes_yaml):
    """a.txt 里只有 人员 和 三轮车。模型说了「卡车」和「三轮车」，卡车是编的。"""
    out = _score(_describe_rows(), classes_yaml, labels_dir=str(labels))
    assert out.loc[0, "chair_available"] == 1
    assert out.loc[0, "mentioned_classes"] == "卡车,三轮车"
    assert out.loc[0, "hallucinated_classes"] == "卡车"
    assert out.loc[0, "chair_s"] == 1
    assert out.loc[0, "chair_i"] == 0.5


def test_declared_image_classes_takes_priority_over_the_annotation_file(labels, classes_yaml):
    out = _score(_describe_rows(**{"meta.image_classes": ["卡车", "三轮车"]}),
                 classes_yaml, labels_dir=str(labels))
    assert out.loc[0, "hallucinated_classes"] == ""    # 两个都在声明的集合里
    assert out.loc[0, "chair_s"] == 0


def test_inventory_is_the_last_resort_when_no_labels_dir_is_given(classes_yaml):
    out = _score(_describe_rows(**{"meta.inventory": ["卡车x1", "三轮车x2"]}), classes_yaml)
    assert out.loc[0, "chair_available"] == 1
    assert out.loc[0, "chair_s"] == 0


def test_chair_stays_unavailable_when_no_source_can_supply_the_class_set(classes_yaml):
    out = _score(_describe_rows(), classes_yaml)
    assert out.loc[0, "chair_available"] == 0


def test_a_fallback_class_table_still_blocks_chair_even_with_labels_dir(labels):
    """类别表不权威时，模型编出来的词根本不会被识别成类别，幻觉会被漏报 ——
    这跟集合完不完整是两个独立的条件，两个都得满足。"""
    out = _score(_describe_rows(), None, labels_dir=str(labels))
    assert out.loc[0, "chair_available"] == 0
