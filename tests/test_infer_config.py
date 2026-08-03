import json

from eval_tool.config import load_infer_config


def test_load_infer_config_reads_batch_and_gpu_parallel_options(tmp_path):
    config_path = tmp_path / "infer.json"
    config_path.write_text(
        json.dumps(
            {
                "infer": {
                    "model_name": "base",
                    "model_path": "model",
                    "tsv_dir": "tsv",
                    "out_dir": "out",
                    "prompt_files": {"mcq": "prompt.txt"},
                    "batch_size": 4,
                    "gpu_ids": [0, 1],
                    "workers_per_gpu": 2,
                    "resume": True,
                    "clean_partial": True,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_infer_config(config_path)

    assert config.batch_size == 4
    assert config.gpu_ids == [0, 1]
    assert config.workers_per_gpu == 2
    assert config.resume is True
    assert config.clean_partial is True
    assert config.overwrite is None
