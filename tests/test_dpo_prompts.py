from pathlib import Path

import pytest

from eval_tool.dpo_prompts import dpo_prompt_path, load_dpo_prompt


def test_binary_visual_and_binary_text_use_different_prompts():
    visual_path = dpo_prompt_path("binary", True)
    text_path = dpo_prompt_path("binary", False)

    assert visual_path != text_path
    assert load_dpo_prompt("binary", True) != load_dpo_prompt("binary", False)


def test_visual_v4_resolves_to_existing_prompt():
    path = dpo_prompt_path("v4", True)

    assert path.name == "judge_equip_pointwise_v4.txt"
    assert path.is_file()


def test_text_v4_defines_text_only_schema():
    prompt = load_dpo_prompt("v4", False)

    assert "fact_score" in prompt
    assert "必须" in prompt and "0-100" in prompt
    assert "visual_score" in prompt and "null" in prompt
    assert "任务对象或主体" in prompt


def test_text_binary_has_no_must_see_image_criterion():
    prompt = load_dpo_prompt("binary", False).lower()

    assert "图片" not in prompt
    assert "图像" not in prompt
    assert "看图" not in prompt
    assert "视觉" not in prompt
    assert "image" not in prompt


@pytest.mark.parametrize("rubric", ["", "V4", "v1", "binary "])
def test_prompt_routing_rejects_unknown_rubrics(rubric):
    with pytest.raises(ValueError):
        dpo_prompt_path(rubric, False)


def test_prompt_paths_are_absolute_files():
    for rubric in ("binary", "v4"):
        for has_images in (False, True):
            path = dpo_prompt_path(rubric, has_images)
            assert isinstance(path, Path)
            assert path.is_absolute()
            assert path.is_file()
