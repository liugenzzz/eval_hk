import json
from pathlib import Path

import pytest

from eval_tool.config import ConfigError, load_infer_config


def _write_infer(tmp_path, **values):
    config_path = tmp_path / "infer.json"
    config_path.write_text(
        json.dumps(
            {
                "infer": {
                    "model_name": "base",
                    "model_path": "model",
                    **values,
                }
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_infer_example_explicitly_loads_training_image_pixel_bounds():
    example_path = Path(__file__).parents[1] / "infer_config.example.json"
    raw = json.loads(example_path.read_text(encoding="utf-8"))

    assert raw["infer"]["image_min_pixels"] == 65_536
    assert raw["infer"]["image_max_pixels"] == 589_824

    config = load_infer_config(example_path)

    assert config.image_min_pixels == 65_536
    assert config.image_max_pixels == 589_824


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


def test_load_infer_config_does_not_treat_false_strings_as_true(tmp_path):
    config_path = tmp_path / "infer.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "base",
                "model_path": "model",
                "overwrite": "false",
                "resume": "false",
                "clean_partial": "false",
            }
        ),
        encoding="utf-8",
    )

    config = load_infer_config(config_path)

    assert config.overwrite is False
    assert config.resume is False
    assert config.clean_partial is False


def test_load_infer_config_reads_image_pixel_bounds(tmp_path):
    config = load_infer_config(
        _write_infer(
            tmp_path,
            image_min_pixels=65536,
            image_max_pixels=589824,
        )
    )

    assert config.image_min_pixels == 65536
    assert config.image_max_pixels == 589824


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"image_min_pixels": None, "image_max_pixels": None},
    ],
)
def test_load_infer_config_defaults_disabled_image_pixel_bounds_to_none(
    tmp_path, values
):
    config = load_infer_config(_write_infer(tmp_path, **values))

    assert config.image_min_pixels is None
    assert config.image_max_pixels is None


def test_load_infer_config_supports_legacy_uppercase_image_pixel_bounds(tmp_path):
    config = load_infer_config(
        _write_infer(
            tmp_path,
            IMAGE_MIN_PIXELS=65536,
            IMAGE_MAX_PIXELS=589824,
        )
    )

    assert config.image_min_pixels == 65536
    assert config.image_max_pixels == 589824


def test_load_infer_config_prefers_present_lowercase_bounds_over_uppercase(tmp_path):
    path = _write_infer(
        tmp_path,
        image_min_pixels=0,
        image_max_pixels=589824,
        IMAGE_MIN_PIXELS=65536,
        IMAGE_MAX_PIXELS=589824,
    )

    with pytest.raises(ConfigError, match=r"image_min_pixels.*> 0"):
        load_infer_config(path)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"image_min_pixels": 65536}, r"image_min_pixels.*image_max_pixels.*together"),
        (
            {"image_min_pixels": True, "image_max_pixels": 589824},
            r"image_min_pixels.*int",
        ),
        (
            {"image_min_pixels": 65536.0, "image_max_pixels": 589824},
            r"image_min_pixels.*int",
        ),
        (
            {"image_min_pixels": "65536", "image_max_pixels": 589824},
            r"image_min_pixels.*int",
        ),
        (
            {"image_min_pixels": 65536, "image_max_pixels": 0},
            r"image_max_pixels.*> 0",
        ),
        (
            {"image_min_pixels": -1, "image_max_pixels": 589824},
            r"image_min_pixels.*> 0",
        ),
        (
            {"image_min_pixels": 589825, "image_max_pixels": 589824},
            r"image_min_pixels.*<=.*image_max_pixels",
        ),
    ],
)
def test_load_infer_config_rejects_invalid_image_pixel_bounds(
    tmp_path, values, message
):
    with pytest.raises(ConfigError, match=message):
        load_infer_config(_write_infer(tmp_path, **values))
