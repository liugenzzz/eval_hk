from __future__ import annotations

import json
from pathlib import Path

from eval_tool.dpo_config import load_dpo_config
from eval_tool.dpo_input import deduplicate_candidates, normalize_inputs
from merge_dpo_shards import recover_dpo_shards


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def _write_jsonl(path: Path, values) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values),
        encoding="utf-8",
    )
    return path


def _fixture(tmp_path: Path):
    image = tmp_path / "images" / "a.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"test-image-placeholder")
    alpaca_source = _write_jsonl(
        tmp_path / "alpaca.jsonl",
        [
            {
                "instruction": "问题一",
                "output": "标准一",
                "image": "images/a.png",
            },
        ],
    )
    sharegpt_source = _write_jsonl(
        tmp_path / "sharegpt.jsonl",
        [
            {
                "conversations": [
                    {"from": "human", "value": "问题二"},
                    {"from": "gpt", "value": "标准二"},
                ]
            }
        ],
    )
    model = tmp_path / "model"
    model.mkdir()
    config_path = _write_json(
        tmp_path / "dpo_full.json",
        {
            "inputs": [str(alpaca_source), str(sharegpt_source)],
            "output_dir": "out",
            "output_name": "train_dpo.jsonl",
            "work_dir": "work",
            "image_root": ".",
            "wrong_only": True,
            "rubric": "v4",
            "infer": {
                "model_name": "unused",
                "model_path": str(model),
                "enable_thinking": False,
                "max_new_tokens": 32,
                "batch_size": 1,
                "torch_dtype": "auto",
                "device_map": "auto",
                "gpu_ids": [],
                "workers_per_gpu": 1,
            },
            "judge": {
                "api_base": "http://127.0.0.1:8000/v1/chat/completions",
                "api_key": "unused",
                "model": "unused",
                "temperature": 0.0,
                "timeout": 120,
                "max_retries": 3,
                "max_workers": 8,
            },
        },
    )
    config = load_dpo_config(config_path)
    normalized = normalize_inputs(config.inputs, image_root=config.image_root)
    candidates = deduplicate_candidates(normalized.candidates).candidates

    attempt_id = "a" * 32
    phase_dir = config.work_dir / "inference"
    attempt_dir = phase_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    _write_json(
        phase_dir / "active.json",
        {"attempt_id": attempt_id, "expected_id_count": len(candidates)},
    )
    records = [
        {
            "sample_id": candidates[1].sample_id,
            "status": "ok",
            "prediction": "模型二",
            "error_type": None,
            "error_message": None,
        },
        {
            "sample_id": candidates[0].sample_id,
            "status": "ok",
            "prediction": "模型一",
            "error_type": None,
            "error_message": None,
        },
    ]
    (attempt_dir / "worker-000000.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    return config_path


def test_recovers_strict_training_rows_in_source_order(tmp_path):
    config_path = _fixture(tmp_path)

    result = recover_dpo_shards(config_path)

    rows = [
        json.loads(line)
        for line in result.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert result.shard_record_count == 2
    assert result.training_row_count == 2
    assert [row["chosen"] for row in rows] == [
        {"from": "gpt", "value": "标准一"},
        {"from": "gpt", "value": "标准二"},
    ]
    assert [row["rejected"] for row in rows] == [
        {"from": "gpt", "value": "模型一"},
        {"from": "gpt", "value": "模型二"},
    ]
    assert [row["images"] for row in rows] == [["images/a.png"], []]
    assert rows[0]["conversations"][0]["value"].startswith("<image>\n")
    assert all(
        set(row) == {"conversations", "chosen", "rejected", "images"}
        for row in rows
    )
