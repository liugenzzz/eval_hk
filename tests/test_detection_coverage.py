import json

from tools.detection_coverage import audit


def test_coverage_counts_turns_and_missing_weighted_slices(tmp_path):
    path = tmp_path / "det.jsonl"
    records = [
        {"metadata": {"task_type": "ground_full"}, "conversations": [
            {"from": "human", "value": "where"}, {"from": "gpt", "value": "box"},
            {"from": "human", "value": "describe"}, {"from": "gpt", "value": "text"}]},
        {"metadata": {"task_type": "attribute_qa"}, "conversations": [
            {"from": "human", "value": "what"}, {"from": "gpt", "value": "blue"}]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    result = audit(path)
    assert result["records"] == 2
    assert result["task_counts"] == {"attribute_qa": 1, "ground_full": 1}
    assert result["scoring_slices"]["ground_box"]["rows"] == 1
    assert result["scoring_slices"]["describe"]["rows"] == 1
    assert result["scoring_slices"]["attribute_qa"]["rows"] == 1
    assert result["scoring_slices"]["detect_box"]["rows"] == 0
    assert result["scoring_slices"]["count_class"]["rows"] == 0
    assert result["scoring_slices"]["describe_modelhist"]["source"] == "derived_at_runtime"
    assert result["scoring_slices"]["describe_modelhist"]["rows"] is None
    assert result["acceptance_coverage"] == 0.45
