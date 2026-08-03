import pytest

from eval_tool.config import EvalConfig, ModelConfig
from eval_tool.rubrics import RubricError, apply_rubric, rubric_prompt_path


def _eval_config(tmp_path):
    return EvalConfig(
        tsv_dir=tmp_path / "tsv",
        out_dir=tmp_path / "reports" / "eval_report",
        cache_dir=tmp_path / "cache",
        datasets={"vqa": "aero_vqa"},
        models=[ModelConfig(name="base")],
        do_pairwise=True,
    )


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v3b", "v4", "v4b"])
def test_rubric_registry_points_to_existing_prompts(version):
    assert rubric_prompt_path(version).is_file()


def test_apply_rubric_changes_prompt_fingerprint_and_output(tmp_path):
    config = _eval_config(tmp_path)

    derived = apply_rubric(config, "v4")

    assert derived.out_dir == tmp_path / "reports" / "eval_report_v4"
    assert derived.judge.pointwise_prompt != config.judge.pointwise_prompt
    assert derived.judge.fingerprint != config.judge.fingerprint
    assert derived.judge.pairwise_prompt == config.judge.pairwise_prompt
    assert config.out_dir == tmp_path / "reports" / "eval_report"


def test_apply_rubric_supports_out_root_and_no_pairwise(tmp_path):
    config = _eval_config(tmp_path)

    derived = apply_rubric(
        config,
        "v3b",
        out_root=tmp_path / "custom",
        no_pairwise=True,
    )

    assert derived.out_dir == tmp_path / "custom" / "eval_report_v3b"
    assert derived.do_pairwise is False
    assert config.do_pairwise is True


def test_unknown_rubric_is_rejected(tmp_path):
    with pytest.raises(RubricError, match="unknown rubric: bad"):
        apply_rubric(_eval_config(tmp_path), "bad")
