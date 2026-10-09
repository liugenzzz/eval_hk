"""Eval inference decodes greedily unless the config asks for sampling.

Every Qwen3-VL checkpoint here ships generation_config do_sample=true / temperature
0.7. Left alone, the same checkpoint scores differently run to run and a gap
between two checkpoints carries sampling noise.
"""

import json

import pytest

from eval_tool.config import ConfigError, load_pipeline_config
from eval_tool.infer_cache import build_infer_fingerprint


def _config(tmp_path, **infer):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({
        "profile": {"name": "aircraft_maintenance_v2"},
        "tsv_dir": str(tmp_path),
        "models": [{"name": "m", "model_path": str(tmp_path)}],
        "baseline_model": "m",
        "infer": dict(infer),
    }), encoding="utf-8")
    return path


def test_greedy_by_default_and_passed_to_every_model(tmp_path):
    config = load_pipeline_config(_config(tmp_path))
    assert config.infer.generation == {"do_sample": False}
    infer_configs = config.to_infer_configs()
    assert infer_configs and all(c.generation == {"do_sample": False} for c in infer_configs)


def test_sampling_must_be_asked_for(tmp_path):
    config = load_pipeline_config(_config(tmp_path, generation={"do_sample": True, "temperature": 0.7}))
    assert config.infer.generation == {"do_sample": True, "temperature": 0.7}


def test_unknown_generation_key_is_rejected(tmp_path):
    with pytest.raises(ConfigError):
        load_pipeline_config(_config(tmp_path, generation={"temprature": 0.7}))


def test_decoding_is_part_of_the_inference_fingerprint():
    args = ("m", "{question}", 512, "k", "d", [{"index": "1"}])
    greedy = build_infer_fingerprint(*args, generation={"do_sample": False})
    sampled = build_infer_fingerprint(*args, generation={"do_sample": True, "temperature": 0.7})
    assert greedy != sampled
    # callers that do not pass generation (the DPO path) keep their old fingerprints
    assert build_infer_fingerprint(*args) == build_infer_fingerprint(*args, generation=None)
