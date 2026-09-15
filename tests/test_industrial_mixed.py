"""A single mixed JSONL must be evaluated once, not twice per language."""

import json
from pathlib import Path

import pytest

from eval_tool.config import load_pipeline_config
from eval_tool.eval_set import load_eval_set, selection_matches


ROOT = Path(__file__).resolve().parents[1]


def test_mixed_profile_slices_one_file_into_disjoint_language_rows(tmp_path):
    config = load_pipeline_config(ROOT / "industrial_mixed.example.json")
    assert set(config.datasets.values()) == {"industrial_eval"}
    assert not config.do_pairwise
    assert set(config.dataset_kinds.values()) == {"industrial_describe"}
    path = tmp_path / "industrial_eval.jsonl"
    records = [
        {"id": f"{lang}_{i}", "images": ["/code/images/a.png"],
         "conversations": [{"from": "human", "value": "<image>describe"},
                           {"from": "gpt", "value": "reference"}],
         "metadata": {"language": lang, "object_category": "aircraft", "defect_type": "crack"}}
        for lang in ("zh", "en") for i in range(3)
    ]
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    frames = {}
    for lang in ("zh", "en"):
        params = config.dataset_params[f"industrial_{lang}"]
        frames[lang] = load_eval_set(path, select=params["select"], category_field="object_category")
        assert len(frames[lang]) == 3
        assert set(frames[lang]["meta.language"]) == {lang}
        assert set(frames[lang]["category"]) == {"aircraft"}
    assert set(frames["zh"]["index"]).isdisjoint(frames["en"]["index"])
    infer_config = config.to_infer_configs([config.models[0].name])[0]
    assert infer_config.pred_stems["industrial_zh"] != infer_config.pred_stems["industrial_en"]


@pytest.mark.parametrize("rule,wanted", [
    ({"language": "zh"}, True), ({"language": "en"}, False),
    ({"language": ["en", "zh"], "turn": 1}, True),
    ({"language": "zh", "turn": 2}, False),
    ({"task_type": ["describe"], "turn": 1}, True),
])
def test_language_filter_combines_with_existing_rules(rule, wanted):
    assert selection_matches({"meta.language": "zh", "turn": 1, "task_type": "describe"}, [rule]) is wanted


def test_missing_language_never_matches_a_language_slice():
    assert not selection_matches({"turn": 1}, [{"language": "zh"}])
    assert selection_matches({"turn": 1}, [{"turn": 1}])


def test_real_training_shape_extracts_and_loads_into_two_language_slices(tmp_path):
    from PIL import Image
    from eval_tool.industrial_extract import extract
    from eval_tool.io import load_truth_dataset

    image_dir = tmp_path / "defect-detect-images" / "0-4aircraft4000"
    image_dir.mkdir(parents=True)
    images = []
    for i in range(3):
        path = image_dir / f"crack_{i}.png"
        Image.new("RGB", (8, 8), "white").save(path)
        images.append(str(path))
    inputs = {}
    for language in ("zh", "en"):
        answer = ("类别：飞机；异常类型：裂纹；描述：机身表面有裂纹。" if language == "zh"
                  else "Category: aircraft; Anomaly type: crack; Description: A crack on the skin.")
        records = [{"conversations": [{"from": "human", "value": "<image>describe"},
                                      {"from": "gpt", "value": answer}], "images": [path]}
                   for path in images]
        inputs[language] = tmp_path / f"merged_{language}.jsonl"
        inputs[language].write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records),
                                    encoding="utf-8")
    output = tmp_path / "sample"
    manifest = extract(inputs["zh"], inputs["en"], output, zh_n=2, en_n=2)
    assert manifest["selected_count"] == 4
    config = load_pipeline_config(ROOT / "industrial_mixed.example.json")
    all_indices = set()
    for language in ("zh", "en"):
        params = {**config.dataset_params[f"industrial_{language}"], "image_root": str(tmp_path)}
        rows = load_truth_dataset(output, "industrial_eval", params)
        assert len(rows) == 2
        assert set(rows["category"]) == {"aircraft"}
        assert set(rows["meta.defect_type"]) == {"crack"}
        assert set(rows["meta.language"]) == {language}
        assert rows["image"].str.len().min() > 0
        assert set(rows["index"]).isdisjoint(all_indices)
        all_indices.update(rows["index"])
        assert rows["answer"].str.startswith("类别" if language == "zh" else "Category").all()
