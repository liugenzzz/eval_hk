import json

import pytest

import eval_tool.industrial_extract as module
from eval_tool.industrial_extract import extract


def record(image, lang="en", question="inspect", obj=None, defect=None):
    obj = obj or ("飞机" if lang == "zh" else "aircraft")
    defect = defect or ("裂纹" if lang == "zh" else "crack")
    answer = f"类别：{obj}；异常类型：{defect}；描述：表面裂纹" if lang == "zh" else f"Category: {obj}; Anomaly type: {defect}; Description: visible crack"
    return {"images": [f"/code/defect-detect-images/0-aircraft/{image}.jpg"],
            "conversations": [{"from": "human", "value": "<image>" + question}, {"from": "gpt", "value": answer}]}


def write(path, records):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return path


def test_mixed_unique_images_and_preserved_records(tmp_path):
    zh = [record(i, "zh", str(q)) for i in range(5) for q in range(3)]
    en = [record(i, "en", str(q)) for i in range(7) for q in range(2)]
    zh[0]["category"] = "旧标签"
    for r in zh:
        r["metadata"] = {"category": "错误旧标签", "custom": 3}
    paths = [write(tmp_path / "zh.jsonl", zh), write(tmp_path / "en.jsonl", en)]
    manifest = extract(*paths, tmp_path / "out", 5, 6)
    output = [json.loads(line) for line in (tmp_path / "out/industrial_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(output) == 11 and len({r["id"] for r in output}) == 11
    assert manifest["shared_bilingual_images"] >= 4
    for lang, count in (("zh", 5), ("en", 6)):
        selected = [r for r in output if r["metadata"]["language"] == lang]
        assert len({r["metadata"]["source_id"] for r in selected}) == count
        assert all(r["metadata"]["object_category"] == "aircraft" for r in selected)
        assert all(r["metadata"]["category"] == "aircraft" for r in selected)
        assert all(r["metadata"]["source_image"] == r["metadata"]["source_image_key"] for r in selected)
        assert all(r["conversations"] in [original["conversations"] for original in (zh if lang == "zh" else en)] for r in selected)
    write(paths[0], list(reversed(zh)))
    write(paths[1], list(reversed(en)))
    shuffled = extract(*paths, tmp_path / "reordered", 5, 6)
    assert shuffled["selected_ids"] == manifest["selected_ids"]
    with pytest.raises(FileExistsError):
        extract(*paths, tmp_path / "out", 5, 6)


def test_unknown_and_conflict_fail_with_audit(tmp_path):
    zh = [record(1, "zh"), record(2, "zh", obj="未知物品"), record(3, "zh")]
    en = [record(1), record(3), record(3, defect="dent")]
    paths = [write(tmp_path / "zh.jsonl", zh), write(tmp_path / "en.jsonl", en)]
    with pytest.raises(ValueError, match="Insufficient"):
        extract(*paths, tmp_path / "out", 3, 3)
    assert not (tmp_path / "out/industrial_eval.jsonl").exists()
    manifest = json.loads((tmp_path / "out/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "insufficient_samples"
    assert manifest["rejected_records"] >= 3
    assert (tmp_path / "out/mapping_audit.json").exists()


def test_malformed_and_nonpaired_languages(tmp_path):
    zh = [record(1, "zh"), record(2, "zh"), {"images": []}]
    en = [record(1), record(3), record(4)]
    paths = [write(tmp_path / "zh.jsonl", zh), write(tmp_path / "en.jsonl", en)]
    manifest = extract(*paths, tmp_path / "out", 2, 3)
    assert manifest["selected_count"] == 5
    assert manifest["rejected_records"] == 1


def test_synonym_mapping_override(tmp_path):
    zh = [record(9, "zh", obj="机体", defect="开裂")]
    en = [record(1)]
    paths = [write(tmp_path / "zh.jsonl", zh), write(tmp_path / "en.jsonl", en)]
    mapping = tmp_path / "mapping.json"
    mapping.write_text(json.dumps({"objects": [{"source_domain": "0-aircraft", "alias": "机体", "canonical": "aircraft"}], "defects": [{"source_domain": "0-aircraft", "object_category": "aircraft", "alias": "开裂", "canonical": "crack"}]}), encoding="utf-8")
    manifest = extract(*paths, tmp_path / "out", 1, 1, mapping=mapping)
    assert manifest["shared_bilingual_images"] == 0
    assert manifest["selected_count"] == 2


def test_object_then_defect_balancing(tmp_path):
    en = [record(f"{obj}-{defect}-{i}", obj=obj, defect=defect)
          for obj in ("aircraft", "metal") for defect in ("crack", "dent") for i in range(8)]
    zh = [record(r["images"][0].split("/")[-1][:-4], "zh",
                 obj="飞机" if "aircraft-" in r["images"][0] else "金属",
                 defect="裂纹" if "-crack-" in r["images"][0] else "凹痕") for r in en]
    paths = [write(tmp_path / "zh.jsonl", zh), write(tmp_path / "en.jsonl", en)]
    manifest = extract(*paths, tmp_path / "out", 8, 8)
    assert len(manifest["distribution"]) == 8
    assert all(row["count"] == 2 for row in manifest["distribution"])


@pytest.mark.parametrize("hide_stat_change", [False, True])
def test_source_change_before_selected_read(tmp_path, monkeypatch, hide_stat_change):
    paths = [write(tmp_path / "zh.jsonl", [record(1, "zh")]),
             write(tmp_path / "en.jsonl", [record(1)])]
    original_select = module._select
    changed = False

    def mutate_after_select(*args):
        nonlocal changed
        result = original_select(*args)
        if not changed:
            write(paths[0], [record(1, "zh", question="changed question")])
            changed = True
        return result

    if hide_stat_change:
        monkeypatch.setattr(module, "_signature", lambda path: (123, 456))
    monkeypatch.setattr(module, "_select", mutate_after_select)
    with pytest.raises(ValueError, match="Source changed"):
        extract(*paths, tmp_path / "out", 1, 1)
    assert not (tmp_path / "out/industrial_eval.jsonl").exists()
    assert not (tmp_path / "out/manifest.json").exists()


def test_source_change_during_scan(tmp_path, monkeypatch):
    paths = [write(tmp_path / "zh.jsonl", [record(1, "zh")]),
             write(tmp_path / "en.jsonl", [record(1)])]
    original_read = module._read

    def mutate_during_read(path, *args):
        result = original_read(path, *args)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        return result

    monkeypatch.setattr(module, "_read", mutate_during_read)
    with pytest.raises(ValueError, match="Source changed during scan"):
        extract(*paths, tmp_path / "out", 1, 1)
    assert not (tmp_path / "out").exists()
