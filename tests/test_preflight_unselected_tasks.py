import json

from eval_tool.config import load_pipeline_config
from eval_tool.preflight import preflight, render


def test_check_names_task_types_no_slice_selects(tmp_path):
    rows = [
        {"id": "a", "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "正常"}],
         "metadata": {"task_type": "discrimination", "image_status": "normal"}},
        {"id": "b", "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "3"}],
         "metadata": {"task_type": "brand_new_task"}},
    ]
    (tmp_path / "aircraft_eval.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    config = tmp_path / "aircraft.json"
    config.write_text(json.dumps({
        "profile": {"name": "aircraft_maintenance_v2"},
        "tsv_dir": str(tmp_path),
        "models": [{"name": "m", "model_path": str(tmp_path)}],
        "baseline_model": "m",
        "infer": {},
    }), encoding="utf-8")
    text = render(preflight(load_pipeline_config(config)))
    assert "aircraft_eval 里没有切片选中的任务" in text
    assert "brand_new_task 1 条" in text
    assert "discrimination" not in text.split("没有切片选中的任务")[1].splitlines()[0]
