"""A config kept outside the repo must still find the prompts its profile names.

Profiles write their prompts repo-relative ("prompts/infer_vqa_raw.txt"), but config
paths resolve against the config file's own directory. Before the fix only configs
sitting at the repo root worked; one kept next to its outputs loaded every model and
then died on the first prompt read - and ``check`` had passed it.
"""

import json

from eval_tool.config import REPO_DIR, load_pipeline_config
from eval_tool.preflight import preflight, render


def _config(tmp_path, **infer):
    path = tmp_path / "elsewhere" / "aircraft.json"
    path.parent.mkdir()
    path.write_text(json.dumps({
        "profile": {"name": "aircraft_maintenance_v2"},
        "tsv_dir": str(tmp_path),
        "models": [{"name": "m", "model_path": str(tmp_path)}],
        "baseline_model": "m",
        "infer": dict(infer),
    }), encoding="utf-8")
    return path


def test_profile_prompts_resolve_against_the_repo_not_the_config(tmp_path):
    config = load_pipeline_config(_config(tmp_path))
    prompts = set(config.infer.prompt_files.values())
    assert prompts == {REPO_DIR / "prompts" / "infer_vqa_raw.txt"}
    assert all(path.is_file() for path in prompts)


def test_a_prompt_written_in_the_config_still_resolves_next_to_it(tmp_path):
    path = _config(tmp_path, prompt_file="my_prompt.txt")
    config = load_pipeline_config(path)
    assert set(config.infer.prompt_files.values()) == {(path.parent / "my_prompt.txt").resolve()}


def test_check_reports_a_missing_inference_prompt_before_any_model_loads(tmp_path):
    result = preflight(load_pipeline_config(_config(tmp_path, prompt_file="my_prompt.txt")))
    assert result.errors >= 1
    assert "my_prompt.txt  ← 找不到" in render(result)
