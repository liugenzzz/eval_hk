import json

import pytest

from eval_tool.industrial_sample import sample_paired
from eval_tool.eval_set import load_eval_set


def record(i, language="zh"):
    return {"id": f"{language}-{i}", "images": [f"images/{i}.png"],
            "metadata": {"pair_id": str(i), "object_category": "metal",
                         "defect_type": "scratch" if i % 2 else "dent"},
            "messages": [{"role": "user", "content": f"Question {language} {i}"},
                         {"role": "assistant", "content": f"Answer {language} {i}"}]}


def write(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def sources(tmp_path, count=10):
    return (write(tmp_path / "zh.jsonl", [record(i) for i in range(count)]),
            write(tmp_path / "en.jsonl", [record(i, "en") for i in range(count)]))


def test_balanced_paired_deterministic_preservation(tmp_path):
    zh, en = sources(tmp_path)
    manifest = sample_paired(zh, en, tmp_path / "a", 6)
    assert [d["selected"] for d in manifest["distribution"]] == [3, 3]
    for lang in ("zh", "en"):
        output = [json.loads(line) for line in (tmp_path / "a" / f"industrial_{lang}.jsonl").read_text().splitlines()]
        assert [r["metadata"]["pair_id"] for r in output] == manifest["sample_ids"]
        assert len(output) == 6
        for r in output:
            expected = record(int(r["metadata"]["pair_id"]), lang)
            expected["metadata"]["language"] = lang
            expected["metadata"]["original_id"] = expected["id"]
            expected["id"] = r["metadata"]["pair_id"]
            assert r == expected
    write(zh, [record(i) for i in reversed(range(10))])
    write(en, [record(i, "en") for i in reversed(range(10))])
    reordered = sample_paired(zh, en, tmp_path / "b", 6)
    assert reordered["sample_ids"] == manifest["sample_ids"]
    assert reordered["distribution"] == manifest["distribution"]
    assert reordered["sources"]["zh"]["sha256"] != manifest["sources"]["zh"]["sha256"]
    for lang in ("zh", "en"):
        assert (tmp_path / "a" / f"industrial_{lang}.jsonl").read_bytes() == (tmp_path / "b" / f"industrial_{lang}.jsonl").read_bytes()


def test_count_and_no_overwrite(tmp_path):
    zh, en = sources(tmp_path, 3)
    with pytest.raises(ValueError, match="only 3"):
        sample_paired(zh, en, tmp_path / "out", 4)
    assert not (tmp_path / "out").exists()
    sample_paired(zh, en, tmp_path / "out", 3)
    original = (tmp_path / "out" / "industrial_zh.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        sample_paired(zh, en, tmp_path / "out", 2)
    assert (tmp_path / "out" / "industrial_zh.jsonl").read_bytes() == original


@pytest.mark.parametrize("bad", ["missing", "duplicate", "mismatch", "empty", "multiturn", "labels", "metadata", "object"])
def test_bad_data_rejected_before_writing(tmp_path, bad):
    zh, en = sources(tmp_path, 3)
    records = [record(i) for i in range(3)]
    if bad == "missing":
        records[0].pop("id")
        records[0]["metadata"].pop("pair_id")
    elif bad == "duplicate":
        records.append(records[0])
    elif bad == "mismatch":
        records.pop()
    elif bad == "empty":
        records[0]["messages"][1]["content"] = " "
    elif bad == "multiturn":
        records[0]["messages"] *= 2
    elif bad == "labels":
        records[0]["metadata"]["defect_type"] = "different"
    elif bad == "metadata":
        records[0]["metadata"] = []
    elif bad == "object":
        records[0]["metadata"].pop("object_category")
    write(zh, records)
    with pytest.raises(ValueError):
        sample_paired(zh, en, tmp_path / "out", 1)
    assert not (tmp_path / "out").exists()


def test_json_sharegpt_and_missing_labels(tmp_path):
    records = [{"id": i, "images": [f"{i}.png"], "metadata": {"object_category": "metal"}, "conversations": [{"from": "human", "value": "Describe"},
                                            {"from": "gpt", "value": "Scratch"}]} for i in range(2)]
    zh = tmp_path / "zh.json"
    zh.write_text(json.dumps(records), encoding="utf-8")
    en = write(tmp_path / "en.jsonl", records)
    manifest = sample_paired(zh, en, tmp_path / "out", 2)
    assert manifest["distribution"] == [{"object_category": "metal", "defect_type": None,
                                          "source_domain": None, "available": 2, "selected": 2}]


def test_objects_balanced_before_defects(tmp_path):
    records = [record(i) for i in range(12)]
    for i, r in enumerate(records):
        r["metadata"]["object_category"] = "metal" if i < 6 else "wood"
        r["metadata"]["defect_type"] = str(i % 3) if i < 6 else "scratch"
    zh = write(tmp_path / "zh.jsonl", records)
    en = write(tmp_path / "en.jsonl", list(reversed(records)))
    manifest = sample_paired(zh, en, tmp_path / "out", 6)
    assert sum(d["selected"] for d in manifest["distribution"] if d["object_category"] == "metal") == 3
    assert sum(d["selected"] for d in manifest["distribution"] if d["object_category"] == "wood") == 3


def test_pair_id_only_loads_unique_indices(tmp_path):
    records = [record(i) for i in range(3)]
    for r in records:
        r.pop("id")
    zh = write(tmp_path / "zh.jsonl", records)
    en = write(tmp_path / "en.jsonl", records)
    sample_paired(zh, en, tmp_path / "out", 3)
    frame = load_eval_set(tmp_path / "out" / "industrial_zh.jsonl")
    assert set(frame["index"]) == {"0__t1", "1__t1", "2__t1"}
    assert frame["index"].is_unique


@pytest.mark.parametrize("bad", ["no_image", "empty_image", "different_image", "top_category", "original_id"])
def test_image_and_identity_conflicts(tmp_path, bad):
    zh, en = sources(tmp_path, 3)
    records = [record(i) for i in range(3)]
    if bad == "no_image":
        records[0].pop("images")
    elif bad == "empty_image":
        records[0]["images"] = [""]
    elif bad == "different_image":
        records[0]["images"] = ["other.png"]
    elif bad == "top_category":
        records[0]["object_category"] = "other"
    elif bad == "original_id":
        records[0]["metadata"]["original_id"] = "other"
    write(zh, records)
    with pytest.raises(ValueError):
        sample_paired(zh, en, tmp_path / "out", 1)
    assert not (tmp_path / "out").exists()


def test_image_string_and_list_pair_equivalence(tmp_path):
    zh, en = sources(tmp_path, 1)
    r = record(0, "en")
    r["image"] = r.pop("images")[0]
    write(en, [r])
    sample_paired(zh, en, tmp_path / "out", 1)


@pytest.mark.parametrize("tag_count", [1, 3])
def test_reject_image_marker_truncation_or_overrun(tmp_path, tag_count):
    records = [record(0)]
    records[0]["images"] = ["a.png", "b.png"]
    records[0]["messages"][0]["content"] = "<image>" * tag_count + "Describe"
    zh = write(tmp_path / "zh.jsonl", records)
    en = write(tmp_path / "en.jsonl", records)
    with pytest.raises(ValueError, match="image marker count"):
        sample_paired(zh, en, tmp_path / "out", 1)
    assert not (tmp_path / "out").exists()


def test_zero_or_full_image_markers_use_same_all_images(tmp_path):
    r = record(0)
    r["images"] = ["a.png", "b.png"]
    zh = write(tmp_path / "zh.jsonl", [r])
    r["messages"][0]["content"] = "<image><image>Describe"
    en = write(tmp_path / "en.jsonl", [r])
    sample_paired(zh, en, tmp_path / "out", 1)
    for lang in ("zh", "en"):
        frame = load_eval_set(tmp_path / "out" / f"industrial_{lang}.jsonl")
        assert frame.iloc[0]["image_files"] == "a.png,b.png"
