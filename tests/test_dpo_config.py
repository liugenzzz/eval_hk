from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import get_args

import pytest

from eval_tool.config import ConfigError
from eval_tool.dpo_config import (
    DpoBuildConfig,
    DpoConfigError,
    DpoInferConfig,
    DpoJudgeConfig,
    RubricName,
    load_dpo_config,
)


VALID_JUDGE = {
    "api_base": "http://judge.test/v1/chat/completions",
    "api_key": "fixture-secret",
    "model": "fixture-judge",
    "temperature": 0.25,
    "timeout": 30,
    "max_retries": 2,
    "max_workers": 3,
}

INVALID_PATH_VALUES = [
    pytest.param("bad\x00name", id="nul"),
    pytest.param("x" * 40_000, id="too-long"),
]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _valid_raw(config_dir: Path, *, wrong_only: bool = False) -> dict[str, object]:
    input_path = config_dir / "data" / "records.jsonl"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text('{"instruction":"question","output":"answer"}\n', encoding="utf-8")

    model_path = config_dir / "models" / "local-model"
    model_path.mkdir(parents=True, exist_ok=True)
    (config_dir / "publish").mkdir(exist_ok=True)
    (config_dir / "work").mkdir(exist_ok=True)

    raw: dict[str, object] = {
        "inputs": ["data/records.jsonl"],
        "output_dir": "publish",
        "output_name": "training.jsonl",
        "work_dir": "work",
        "wrong_only": wrong_only,
        "infer": {
            "model_name": "fixture-model",
            "model_path": "models/local-model",
        },
    }
    if wrong_only:
        raw["rubric"] = "binary"
        raw["judge"] = dict(VALID_JUDGE)
    return raw


def _write_config(
    tmp_path: Path,
    *,
    wrong_only: bool = False,
) -> tuple[Path, Path, dict[str, object]]:
    config_dir = tmp_path / "configuration"
    config_dir.mkdir(parents=True)
    raw = _valid_raw(config_dir, wrong_only=wrong_only)
    config_path = config_dir / "dpo.json"
    _write_json(config_path, raw)
    return config_path, config_dir, raw


def test_load_dpo_config_resolves_config_paths_and_cwd_image_root(tmp_path):
    config_path, config_dir, _ = _write_config(tmp_path)
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()

    config = load_dpo_config(config_path, invocation_dir=invocation_dir)

    assert isinstance(config, DpoBuildConfig)
    assert config.config_path == config_path.resolve()
    assert config.inputs == ((config_dir / "data/records.jsonl").resolve(),)
    assert config.output_dir == (config_dir / "publish").resolve()
    assert config.work_dir == (config_dir / "work").resolve()
    assert config.image_root == invocation_dir.resolve()
    assert config.output_name == "training.jsonl"
    assert config.wrong_only is False
    assert config.rubric is None
    assert config.judge is None
    assert config.infer == DpoInferConfig(
        model_name="fixture-model",
        model_path=(config_dir / "models/local-model").resolve(),
    )


def test_dpo_example_explicitly_loads_training_image_pixel_bounds(tmp_path):
    example_path = Path(__file__).parents[1] / "dpo.example.json"
    raw = json.loads(example_path.read_text(encoding="utf-8"))

    assert raw["infer"]["image_min_pixels"] == 65_536
    assert raw["infer"]["image_max_pixels"] == 589_824

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "alpaca_train.json").write_text("[]", encoding="utf-8")
    (tmp_path / "data" / "sharegpt_train.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "models" / "Qwen2.5-VL-7B-Instruct").mkdir(parents=True)
    loadable_example = tmp_path / "dpo.example.json"
    _write_json(loadable_example, raw)

    config = load_dpo_config(loadable_example, invocation_dir=tmp_path)

    assert config.infer.image_min_pixels == 65_536
    assert config.infer.image_max_pixels == 589_824


def test_explicit_image_root_resolves_from_config_directory(tmp_path):
    config_path, config_dir, raw = _write_config(tmp_path)
    (config_dir / "assets").mkdir()
    raw["image_root"] = "assets"
    _write_json(config_path, raw)
    invocation_dir = tmp_path / "elsewhere"
    invocation_dir.mkdir()

    config = load_dpo_config(config_path, invocation_dir=invocation_dir)

    assert config.image_root == (config_dir / "assets").resolve()
    assert config.image_root != invocation_dir.resolve()


def test_cli_inputs_replace_config_inputs_and_resolve_from_invocation_dir(tmp_path):
    config_path, config_dir, _ = _write_config(tmp_path)
    invocation_dir = tmp_path / "invocation"
    invocation_dir.mkdir()
    override = invocation_dir / "override.jsonl"
    override.write_text("{}\n", encoding="utf-8")

    config = load_dpo_config(
        config_path,
        input_overrides=["override.jsonl"],
        invocation_dir=invocation_dir,
    )

    assert config.inputs == (override.resolve(),)
    assert (config_dir / "data/records.jsonl").resolve() not in config.inputs


@pytest.mark.parametrize(
    ("field", "error_field"),
    [
        ("inputs", r"inputs\[0\]"),
        ("model_path", "infer.model_path"),
        ("output_dir", "output_dir"),
        ("work_dir", "work_dir"),
        ("image_root", "image_root"),
    ],
)
@pytest.mark.parametrize("invalid_path", INVALID_PATH_VALUES)
def test_config_path_fields_normalize_filesystem_errors(
    tmp_path, field, error_field, invalid_path
):
    config_path, _, raw = _write_config(tmp_path)
    if field == "inputs":
        raw["inputs"] = [invalid_path]
    elif field == "model_path":
        infer = raw["infer"]
        assert isinstance(infer, dict)
        infer["model_path"] = invalid_path
    else:
        raw[field] = invalid_path
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match=error_field):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize("invalid_path", INVALID_PATH_VALUES)
def test_cli_input_path_errors_are_normalized(tmp_path, invalid_path):
    config_path, _, _ = _write_config(tmp_path)

    with pytest.raises(DpoConfigError, match=r"inputs\[0\]"):
        load_dpo_config(
            config_path,
            input_overrides=[invalid_path],
            invocation_dir=tmp_path,
        )


@pytest.mark.parametrize("invalid_path", INVALID_PATH_VALUES)
def test_config_path_argument_errors_are_normalized(tmp_path, invalid_path):
    with pytest.raises(DpoConfigError, match="config_path"):
        load_dpo_config(invalid_path, invocation_dir=tmp_path)


@pytest.mark.parametrize("invalid_path", INVALID_PATH_VALUES)
def test_invocation_dir_errors_are_normalized(tmp_path, invalid_path):
    config_path, _, _ = _write_config(tmp_path)

    with pytest.raises(DpoConfigError, match="invocation_dir"):
        load_dpo_config(config_path, invocation_dir=Path(invalid_path))


@pytest.mark.parametrize("value", [None, "false", 0, 1, [], {}])
def test_wrong_only_requires_literal_json_boolean(tmp_path, value):
    config_path, _, raw = _write_config(tmp_path)
    raw["wrong_only"] = value
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match="wrong_only.*boolean"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


def test_wrong_only_is_required(tmp_path):
    config_path, _, raw = _write_config(tmp_path)
    del raw["wrong_only"]
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match="wrong_only.*required"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize("value", [None, "false", 0, 1, [], {}])
def test_enable_thinking_requires_literal_json_boolean(tmp_path, value):
    config_path, _, raw = _write_config(tmp_path)
    infer = raw["infer"]
    assert isinstance(infer, dict)
    infer["enable_thinking"] = value
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match="infer.enable_thinking.*boolean"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


def test_dpo_infer_loads_image_pixel_bounds(tmp_path):
    config_path, _, raw = _write_config(tmp_path)
    infer = raw["infer"]
    assert isinstance(infer, dict)
    infer.update(
        {
            "image_min_pixels": 65_536,
            "image_max_pixels": 589_824,
        }
    )
    _write_json(config_path, raw)

    config = load_dpo_config(config_path, invocation_dir=tmp_path)

    assert config.infer.image_min_pixels == 65_536
    assert config.infer.image_max_pixels == 589_824


@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param({}, id="omitted"),
        pytest.param(
            {"image_min_pixels": None, "image_max_pixels": None},
            id="explicit-null",
        ),
    ],
)
def test_dpo_infer_disables_image_pixel_bounds_when_omitted_or_null(
    tmp_path, bounds
):
    config_path, _, raw = _write_config(tmp_path)
    infer = raw["infer"]
    assert isinstance(infer, dict)
    infer.update(bounds)
    _write_json(config_path, raw)

    config = load_dpo_config(config_path, invocation_dir=tmp_path)

    assert config.infer.image_min_pixels is None
    assert config.infer.image_max_pixels is None


@pytest.mark.parametrize(
    "bounds",
    [
        pytest.param({"image_min_pixels": 65_536}, id="missing-max"),
        pytest.param({"image_max_pixels": 589_824}, id="missing-min"),
        pytest.param({"image_min_pixels": None}, id="missing-max-with-null-min"),
        pytest.param({"image_max_pixels": None}, id="missing-min-with-null-max"),
        pytest.param(
            {"image_min_pixels": None, "image_max_pixels": 589_824},
            id="null-min",
        ),
        pytest.param(
            {"image_min_pixels": 65_536, "image_max_pixels": None},
            id="null-max",
        ),
    ],
)
def test_dpo_infer_rejects_single_sided_image_pixel_bounds(tmp_path, bounds):
    config_path, _, raw = _write_config(tmp_path)
    infer = raw["infer"]
    assert isinstance(infer, dict)
    infer.update(bounds)
    _write_json(config_path, raw)

    with pytest.raises(
        DpoConfigError,
        match=r"infer.*image_min_pixels.*image_max_pixels",
    ):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize(
    ("field", "invalid", "reason"),
    [
        pytest.param("image_min_pixels", True, "must be an int", id="min-bool"),
        pytest.param(
            "image_min_pixels", 65_536.0, "must be an int", id="min-float"
        ),
        pytest.param(
            "image_min_pixels", "65536", "must be an int", id="min-string"
        ),
        pytest.param("image_min_pixels", 0, "must be > 0", id="min-zero"),
        pytest.param("image_min_pixels", -1, "must be > 0", id="min-negative"),
        pytest.param("image_max_pixels", False, "must be an int", id="max-bool"),
        pytest.param(
            "image_max_pixels", 589_824.0, "must be an int", id="max-float"
        ),
        pytest.param(
            "image_max_pixels", "589824", "must be an int", id="max-string"
        ),
        pytest.param("image_max_pixels", 0, "must be > 0", id="max-zero"),
        pytest.param("image_max_pixels", -1, "must be > 0", id="max-negative"),
    ],
)
def test_dpo_infer_rejects_nonpositive_or_nonstrict_image_pixel_bounds(
    tmp_path, field, invalid, reason
):
    config_path, _, raw = _write_config(tmp_path)
    infer = raw["infer"]
    assert isinstance(infer, dict)
    infer.update({"image_min_pixels": 65_536, "image_max_pixels": 589_824})
    infer[field] = invalid
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match=rf"infer.*{field}.*{reason}"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


def test_dpo_infer_rejects_reversed_image_pixel_bounds(tmp_path):
    config_path, _, raw = _write_config(tmp_path)
    infer = raw["infer"]
    assert isinstance(infer, dict)
    infer.update({"image_min_pixels": 589_824, "image_max_pixels": 65_536})
    _write_json(config_path, raw)

    with pytest.raises(
        DpoConfigError,
        match=r"infer.*image_min_pixels.*<=.*image_max_pixels",
    ):
        load_dpo_config(config_path, invocation_dir=tmp_path)


def test_full_mode_ignores_optional_rubric_and_judge_values(tmp_path):
    config_path, _, raw = _write_config(tmp_path)
    raw["rubric"] = "retired-rubric"
    raw["judge"] = {
        "api_base": "",
        "api_key": "",
        "model": "",
        "temperature": "stale",
        "timeout": False,
        "max_retries": -99,
        "max_workers": 0,
    }
    _write_json(config_path, raw)

    config = load_dpo_config(config_path, invocation_dir=tmp_path)

    assert config.rubric is None
    assert config.judge is None


@pytest.mark.parametrize("rubric", [None, "v1", "V4", 4, True, [], {}])
def test_wrong_only_requires_binary_or_v4_rubric(tmp_path, rubric):
    config_path, _, raw = _write_config(tmp_path, wrong_only=True)
    if rubric is None:
        del raw["rubric"]
    else:
        raw["rubric"] = rubric
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match="rubric.*binary.*v4"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize(
    "missing",
    ["api_base", "api_key", "model", "temperature", "timeout", "max_retries"],
)
def test_wrong_only_requires_complete_judge(tmp_path, missing):
    config_path, _, raw = _write_config(tmp_path, wrong_only=True)
    judge = raw["judge"]
    assert isinstance(judge, dict)
    del judge[missing]
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match=rf"judge.{missing}.*required"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize("judge", [None, [], "judge"])
def test_wrong_only_requires_judge_object(tmp_path, judge):
    config_path, _, raw = _write_config(tmp_path, wrong_only=True)
    raw["judge"] = judge
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match="judge.*object"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


def test_wrong_only_maps_complete_judge_settings_without_output(tmp_path, capsys):
    config_path, _, _ = _write_config(tmp_path, wrong_only=True)

    config = load_dpo_config(config_path, invocation_dir=tmp_path)

    assert config.rubric == "binary"
    assert config.judge == DpoJudgeConfig(
        settings=config.judge.settings,
        max_workers=3,
    )
    assert config.judge.settings.api_base == VALID_JUDGE["api_base"]
    assert config.judge.settings.api_key == VALID_JUDGE["api_key"]
    assert config.judge.settings.model == VALID_JUDGE["model"]
    assert config.judge.settings.temperature == VALID_JUDGE["temperature"]
    assert config.judge.settings.timeout == VALID_JUDGE["timeout"]
    assert config.judge.settings.max_retries == VALID_JUDGE["max_retries"]
    assert capsys.readouterr() == ("", "")


def test_wrong_only_v4_uses_default_judge_max_workers(tmp_path):
    config_path, _, raw = _write_config(tmp_path, wrong_only=True)
    raw["rubric"] = "v4"
    judge = raw["judge"]
    assert isinstance(judge, dict)
    del judge["max_workers"]
    _write_json(config_path, raw)

    config = load_dpo_config(config_path, invocation_dir=tmp_path)

    assert config.rubric == "v4"
    assert config.judge is not None
    assert config.judge.max_workers == 8


def test_dpo_types_are_frozen_and_config_error_inherits_existing_error(tmp_path):
    config_path, _, _ = _write_config(tmp_path, wrong_only=True)
    config = load_dpo_config(config_path, invocation_dir=tmp_path)

    assert issubclass(DpoConfigError, ConfigError)
    assert get_args(RubricName) == ("binary", "v4")
    assert isinstance(config.infer.gpu_ids, tuple)
    with pytest.raises(FrozenInstanceError):
        config.output_name = "changed.jsonl"
    with pytest.raises(FrozenInstanceError):
        config.infer.batch_size = 9
    with pytest.raises(FrozenInstanceError):
        config.judge.max_workers = 9


@pytest.mark.parametrize("missing", ["inputs", "model_path"])
def test_model_path_and_inputs_must_exist(tmp_path, missing):
    config_path, config_dir, raw = _write_config(tmp_path)
    if missing == "inputs":
        raw["inputs"] = ["data/missing.jsonl"]
    else:
        infer = raw["infer"]
        assert isinstance(infer, dict)
        infer["model_path"] = "models/missing"
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match=missing):
        load_dpo_config(config_path, invocation_dir=config_dir)


def test_inputs_must_be_a_nonempty_list_of_files(tmp_path):
    config_path, config_dir, raw = _write_config(tmp_path)
    directory = config_dir / "data" / "directory"
    directory.mkdir()

    for inputs in ([], "data/records.jsonl", ["data/directory"]):
        raw["inputs"] = inputs
        _write_json(config_path, raw)
        with pytest.raises(DpoConfigError, match="inputs"):
            load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize("output_name", ["training", "training.json", "training.jsonl.tmp"])
def test_output_name_must_end_with_jsonl(tmp_path, output_name):
    config_path, _, raw = _write_config(tmp_path)
    raw["output_name"] = output_name
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match="output_name.*jsonl"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize(
    "output_name",
    [
        "../escape.jsonl",
        "nested/output.jsonl",
        r"nested\output.jsonl",
        "/absolute.jsonl",
        "C:drive.jsonl",
        r"C:\absolute.jsonl",
        "two..dots.jsonl",
        "CON.jsonl",
        "nul.jsonl",
        "LPT1.jsonl",
        "COM¹.jsonl",
        "LPT².jsonl",
        "CON .jsonl",
        "NUL .jsonl",
        "CONIN$.jsonl",
        "CONOUT$.jsonl",
        "audit_records.jsonl",
        "rejected_records.jsonl",
        "summary.json",
        "manifest.json",
        "warnings.log",
        "training.jsonl ",
        "training.jsonl.",
    ],
)
def test_output_name_must_be_a_safe_nonreserved_basename(tmp_path, output_name):
    config_path, _, raw = _write_config(tmp_path)
    raw["output_name"] = output_name
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match="output_name"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("infer", "max_new_tokens", True),
        ("infer", "max_new_tokens", 0),
        ("infer", "max_new_tokens", 1.0),
        ("infer", "batch_size", False),
        ("infer", "batch_size", 0),
        ("infer", "workers_per_gpu", True),
        ("infer", "workers_per_gpu", 0),
        ("judge", "timeout", True),
        ("judge", "timeout", 0),
        ("judge", "max_workers", False),
        ("judge", "max_workers", 0),
        ("judge", "max_retries", True),
        ("judge", "max_retries", -1),
        ("judge", "temperature", True),
        ("judge", "temperature", "0.1"),
        ("judge", "temperature", -0.1),
        ("judge", "temperature", 2.1),
        ("judge", "temperature", float("nan")),
        ("judge", "temperature", float("inf")),
        ("judge", "temperature", 10**309),
    ],
)
def test_numeric_fields_reject_bool_and_invalid_ranges(tmp_path, section, field, value):
    config_path, _, raw = _write_config(tmp_path, wrong_only=True)
    settings = raw[section]
    assert isinstance(settings, dict)
    settings[field] = value
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match=rf"{section}.{field}"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize(
    ("location", "unknown"),
    [("top", "wrong_ony"), ("infer", "batch_sze"), ("judge", "timeot")],
)
def test_unknown_top_infer_and_judge_keys_are_rejected(tmp_path, location, unknown):
    config_path, _, raw = _write_config(tmp_path, wrong_only=True)
    if location == "top":
        raw[unknown] = True
    else:
        section = raw[location]
        assert isinstance(section, dict)
        section[unknown] = 1
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match=rf"unknown.*{location}.*{unknown}"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("top", "output_dir"),
        ("top", "work_dir"),
        ("infer", "model_name"),
        ("infer", "model_path"),
        ("infer", "torch_dtype"),
        ("infer", "device_map"),
        ("judge", "api_base"),
        ("judge", "api_key"),
        ("judge", "model"),
    ],
)
def test_required_strings_must_be_nonempty(tmp_path, section, field):
    config_path, _, raw = _write_config(tmp_path, wrong_only=True)
    if section == "top":
        raw[field] = "  "
    else:
        settings = raw[section]
        assert isinstance(settings, dict)
        settings[field] = "  "
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match=field):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize("inputs", [[""], ["  "], [1], [None]])
def test_input_paths_must_be_nonempty_strings(tmp_path, inputs):
    config_path, _, raw = _write_config(tmp_path)
    raw["inputs"] = inputs
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match="inputs"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize("gpu_ids", [[0, 0], [-1], [True], [0.0], "0,1"])
def test_gpu_ids_must_be_a_unique_nonnegative_integer_list(tmp_path, gpu_ids):
    config_path, _, raw = _write_config(tmp_path)
    infer = raw["infer"]
    assert isinstance(infer, dict)
    infer["gpu_ids"] = gpu_ids
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match="infer.gpu_ids"):
        load_dpo_config(config_path, invocation_dir=tmp_path)


@pytest.mark.parametrize("device_map", ["auto", "cuda:0"])
def test_gpu_workers_allow_auto_or_worker_local_cuda_zero(tmp_path, device_map):
    config_path, _, raw = _write_config(tmp_path)
    infer = raw["infer"]
    assert isinstance(infer, dict)
    infer["gpu_ids"] = [2, 7]
    infer["device_map"] = device_map
    _write_json(config_path, raw)

    config = load_dpo_config(config_path, invocation_dir=tmp_path)

    assert config.infer.gpu_ids == (2, 7)
    assert config.infer.device_map == device_map


@pytest.mark.parametrize("device_map", ["cuda:1", "cuda:7"])
def test_gpu_workers_reject_physical_device_numbers(tmp_path, device_map):
    config_path, _, raw = _write_config(tmp_path)
    infer = raw["infer"]
    assert isinstance(infer, dict)
    infer["gpu_ids"] = [2, 7]
    infer["device_map"] = device_map
    _write_json(config_path, raw)

    with pytest.raises(DpoConfigError, match="device_map.*cuda:0"):
        load_dpo_config(config_path, invocation_dir=tmp_path)
