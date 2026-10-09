import json

from eval_tool.config import load_pipeline_config
from eval_tool.preflight import preflight, render


def _setup(tmp_path, table_names):
    row = {"id": "a", "images": [], "conversations": [
        {"from": "human", "value": "框里是什么？"}, {"from": "gpt", "value": "该区域内的是背包。"}],
        "metadata": {"task_type": "region_identify", "label": "背包"}}
    (tmp_path / "test.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    (tmp_path / "classes.yaml").write_text(
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(table_names)), encoding="utf-8")
    config = tmp_path / "det.json"
    config.write_text(json.dumps({
        "profile": {"name": "grounding_zh_v1"},
        "tsv_dir": str(tmp_path),
        "eval_set": "test",
        "enabled_datasets": ["region_identify"],
        "derive": [],
        "dataset_defaults": {"classes_yaml": str(tmp_path / "classes.yaml")},
        "models": [{"name": "m", "model_path": str(tmp_path)}],
        "baseline_model": "m",
        "infer": {},
    }, ensure_ascii=False), encoding="utf-8")
    return render(preflight(load_pipeline_config(config)))


def test_a_class_table_that_misses_the_labels_is_an_error(tmp_path):
    text = _setup(tmp_path, ["Backpack", "Person"])
    assert "✗ 类别表覆盖" in text


def test_a_matching_class_table_passes(tmp_path):
    text = _setup(tmp_path, ["背包", "人员"])
    assert "✓ 类别表覆盖  数据里 100% 的 label 在表里" in text
