import pytest

from eval_tool.industrial_labels import build_label_mapping, extract_labels, image_identity


def row(image="a/1.jpg", obj="aircraft", defect="crack", zh=False):
    answer = (f"类别：{obj}；异常类型：{defect}；描述：原始描述" if zh else
              f"Category: {obj}; Anomaly type: {defect}; Description: Original text")
    return {"conversations": [{"from": "human", "value": "<image>Question"},
                              {"from": "gpt", "value": answer}],
            "images": ["/root/defect-detect-images/" + image]}


def test_order_independent_same_image_anchors_and_preserves_raw():
    en = [row(), row("a/2.jpg", defect="dent")]
    zh = [row("a/2.jpg", "飞机", "凹痕", True), row(obj="飞机", defect="裂纹", zh=True)]
    mapping = build_label_mapping(en, zh)
    result = mapping.resolve(zh[1], "zh")
    assert result["status"] == "resolved"
    assert result["object_category"] == "aircraft"
    assert result["defect_type"] == "crack"
    assert result["raw_object_category"] == "飞机"
    assert "原始描述" in zh[1]["conversations"][1]["value"]


def test_same_image_multiple_questions_are_not_ambiguous():
    zh = row(obj="飞机", defect="裂纹", zh=True)
    mapping = build_label_mapping([row(), row(obj="Airplane")], [zh, zh])
    assert mapping.resolve(zh, "zh")["status"] == "resolved"


def test_conflicting_image_and_conflicting_alias_are_ambiguous():
    zh = row(obj="飞机", defect="裂纹", zh=True)
    mapping = build_label_mapping([row(), row(defect="dent")], [zh])
    assert mapping.resolve(zh, "zh")["status"] == "ambiguous"
    mapping = build_label_mapping([row(), row("a/2.jpg", obj="car")],
                                 [zh, row("a/2.jpg", "飞机", "裂纹", True)])
    assert mapping.resolve(zh, "zh")["status"] == "ambiguous"


def test_defect_alias_scoped_by_object_and_domain():
    en = [row(), row("a/2.jpg", "car", "dent"), row("b/1.jpg", "aircraft", "scratch")]
    zh = [row(obj="飞机", defect="异常", zh=True), row("a/2.jpg", "汽车", "异常", True),
          row("b/1.jpg", "飞机", "异常", True)]
    mapping = build_label_mapping(en, zh)
    assert [mapping.resolve(x, "zh")["defect_type"] for x in zh] == ["crack", "dent", "scratch"]


def test_unmatched_chinese_requires_explicit_override():
    zh = row(obj="飞机", defect="裂纹", zh=True)
    mapping = build_label_mapping([], [zh])
    assert mapping.resolve(zh, "zh")["status"] == "unresolved"
    mapping = build_label_mapping([], [zh], {
        "objects": [{"source_domain": "a", "alias": "飞机", "canonical": "aircraft"}],
        "defects": [{"source_domain": "a", "object_category": "aircraft", "alias": "裂纹", "canonical": "crack"}]})
    assert mapping.resolve(zh, "zh")["status"] == "resolved"
    assert mapping.audit()["manual_overrides"] == {
        "objects": [{"source_domain": "a", "alias": "飞机", "canonical": "aircraft"}],
        "defects": [{"source_domain": "a", "object_category": "aircraft", "alias": "裂纹", "canonical": "crack"}]}


def test_prefix_only_and_image_identity():
    r = row()
    assert extract_labels(r) == ("aircraft", "crack")
    r["conversations"][1]["value"] = "Description: Category: aircraft; Anomaly type: crack; Description: x"
    with pytest.raises(ValueError):
        extract_labels(r)
    assert image_identity(row())[0] == "a/1.jpg"
    r["images"] = ["/a/1.jpg"]
    assert image_identity(r)[0] == "/a/1.jpg"
    r["images"].append("/b/1.jpg")
    with pytest.raises(ValueError):
        image_identity(r)


def test_multiline_prefix_preserves_original_labels():
    r = row(obj="Tungsten\n Inert Gas", defect="good\n weld")
    assert extract_labels(r) == ("Tungsten\n Inert Gas", "good\n weld")
    mapping = build_label_mapping([r], [])
    assert mapping.resolve(r, "en")["object_category"] == "tungsten_inert_gas"


@pytest.mark.parametrize("override", [[], "x", {"object": []}, {"objects": {}},
    {"defects": [None]}, {"objects": [{"source_domain": "a", "alias": "", "canonical": "x"}]},
    {"objects": [{"source_domain": "a", "alias": "x"}]},
    {"defects": [{"source_domain": "a", "object_category": 2, "alias": "x", "canonical": "y"}]}])
def test_invalid_override_schema_rejected(override):
    with pytest.raises(ValueError):
        build_label_mapping([], [], override)


@pytest.mark.parametrize("section", ["objects", "defects"])
def test_conflicting_override_rejected_but_equivalent_duplicate_allowed(section):
    entry = {"source_domain": "a", "alias": "裂纹", "canonical": "crack"}
    if section == "defects":
        entry["object_category"] = "aircraft"
    build_label_mapping([], [], {section: [entry, dict(entry, canonical="CRACK")]})
    with pytest.raises(ValueError, match="Conflicting"):
        build_label_mapping([], [], {section: [entry, dict(entry, canonical="dent")]})
