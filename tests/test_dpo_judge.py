from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from eval_tool.dpo_input import (
    DpoCandidate,
    DpoTurn,
    ImageRef,
    SourceRef,
)
from eval_tool.dpo_judge import (
    build_dpo_judge_request,
    parse_dpo_judge_response,
    validate_dpo_judge_object,
)
from eval_tool.dpo_multimodal import inspect_image
from eval_tool.dpo_prompts import load_dpo_prompt
from eval_tool.judge import JudgeClient, JudgeSettings


def _candidate(
    conversations: tuple[DpoTurn, ...],
    *,
    chosen: str = "标准答案",
    images: tuple[ImageRef, ...] = (),
) -> DpoCandidate:
    return DpoCandidate(
        sample_id="sample-1",
        conversations=conversations,
        chosen=chosen,
        images=images,
        source=SourceRef(
            input_index=0,
            source_path=Path("input.json"),
            container_format="json_array",
            record_index=0,
            line_number=None,
            source_id=None,
            raw_digest="0" * 64,
        ),
        source_format="sharegpt",
        turn_index=1,
        turn_count=2,
    )


def _v4_object(**updates):
    obj = {
        "rubric": "v4",
        "equipment_correct": True,
        "fact_score": 80,
        "param_score": None,
        "visual_score": None,
        "fabrication_score": 100,
        "style_score": 80,
        "reasoning": "ok",
    }
    obj.update(updates)
    return obj


def test_structured_transport_preserves_existing_post_override_compatibility():
    class CompatClient(JudgeClient):
        def __init__(self):
            super().__init__(JudgeSettings())
            self.legacy_calls = []
            self.message_calls = []

        def _post(self, system_prompt, user_text, image_b64=None, mime="image/jpeg"):
            self.legacy_calls.append((system_prompt, user_text, image_b64, mime))
            return '{"correct": true, "reason": "ok"}'

        def _post_messages(self, messages, *, max_tokens=1024):
            self.message_calls.append((messages, max_tokens))
            return "structured"

    client = CompatClient()

    assert client.judge_pointwise("q", "ref", "pred")["hit"] == 1
    assert client.legacy_calls
    assert client.judge_messages([{"role": "user", "content": "q"}], max_tokens=17) == "structured"
    assert client.message_calls == [([{"role": "user", "content": "q"}], 17)]


def test_multiturn_judge_includes_gold_history_current_chosen_and_rejected():
    candidate = _candidate(
        (
            DpoTurn(from_="human", value="历史问题"),
            DpoTurn(from_="gpt", value="历史标准答案"),
            DpoTurn(from_="human", value="当前问题"),
        ),
        chosen="当前标准答案",
    )
    prompt = load_dpo_prompt("binary", False)

    request = build_dpo_judge_request(candidate, "当前模型答案", prompt, {})

    assert request.rubric == "binary"
    assert request.has_images is False
    assert [message["role"] for message in request.messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert request.messages[1]["content"] == "历史问题"
    assert request.messages[2]["content"] == "历史标准答案"
    assert request.messages[3]["content"] == "当前问题"
    comparison = request.messages[4]["content"]
    assert "当前标准答案" in comparison
    assert "当前模型答案" in comparison
    assert "chosen" in comparison
    assert "rejected" in comparison


def test_judge_interleaves_png_and_jpeg_at_original_placeholder_positions(tmp_path):
    png = tmp_path / "first.bin"
    jpeg = tmp_path / "second.bin"
    Image.new("RGB", (2, 2), "red").save(png, format="PNG")
    Image.new("RGB", (2, 2), "blue").save(jpeg, format="JPEG")
    refs = (
        ImageRef(original="relative/first.png", resolved=png.resolve()),
        ImageRef(original="relative/second.jpg", resolved=jpeg.resolve()),
    )
    assets = {ref.resolved: inspect_image(ref) for ref in refs}
    candidate = _candidate(
        (DpoTurn(from_="human", value="前<image>中<image>后"),),
        images=refs,
    )

    request = build_dpo_judge_request(
        candidate,
        "模型答案",
        load_dpo_prompt("binary", True),
        assets,
    )

    content = request.messages[1]["content"]
    assert [part["type"] for part in content] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert content[0]["text"] == "前"
    assert content[2]["text"] == "中"
    assert content[4]["text"] == "后"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[3]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    fingerprint = json.dumps(request.fingerprint_material, ensure_ascii=False)
    assert "data:" not in fingerprint
    assert "base64" not in fingerprint
    assert "relative/first.png" in fingerprint
    assert assets[refs[0].resolved].sha256 in fingerprint


@pytest.mark.parametrize("correct", [True, False])
def test_binary_accepts_only_literal_boolean_correct(correct):
    parsed = parse_dpo_judge_response(
        json.dumps({"correct": correct, "reason": "verdict"}),
        "binary",
        False,
    )

    assert parsed["hit"] == (1 if correct else 0)


@pytest.mark.parametrize("correct", [None, 0, 1, "true", "false", [], {}])
def test_binary_rejects_non_boolean_correct(correct):
    with pytest.raises(ValueError):
        parse_dpo_judge_response(
            json.dumps({"correct": correct, "reason": "bad"}),
            "binary",
            False,
        )


@pytest.mark.parametrize("equipment_correct", [None, 0, 1, "true", "false", [], {}])
def test_v4_accepts_only_literal_boolean_equipment_correct(equipment_correct):
    with pytest.raises(ValueError):
        validate_dpo_judge_object(
            _v4_object(equipment_correct=equipment_correct), "v4", False
        )


@pytest.mark.parametrize("value", [True, False, "50", float("nan"), float("inf"), -1, 101])
@pytest.mark.parametrize(
    "field", [
        "fact_score",
        "param_score",
        "visual_score",
        "fabrication_score",
        "style_score",
    ]
)
def test_v4_numeric_fields_reject_bool_string_nan_and_out_of_range(field, value):
    obj = _v4_object()
    if field == "visual_score":
        has_images = True
        obj["fact_score"] = None
        obj["visual_score"] = value
    else:
        has_images = False
        obj[field] = value

    with pytest.raises(ValueError):
        validate_dpo_judge_object(obj, "v4", has_images)


def test_text_v4_requires_numeric_fact_and_null_visual():
    validate_dpo_judge_object(_v4_object(), "v4", False)

    with pytest.raises(ValueError):
        validate_dpo_judge_object(_v4_object(fact_score=None), "v4", False)
    with pytest.raises(ValueError):
        validate_dpo_judge_object(_v4_object(visual_score=50), "v4", False)


def test_visual_v4_keeps_existing_nullable_dimensions():
    raw = json.dumps(
        _v4_object(fact_score=None, param_score=None, visual_score=75)
    )

    parsed = parse_dpo_judge_response(raw, "v4", True)

    assert parsed["visual_score"] == 75


def test_v4_uses_parser_hit_not_rounded_quality_score():
    raw = json.dumps(
        _v4_object(
            fact_score=59.999,
            param_score=None,
            visual_score=None,
            fabrication_score=100,
        )
    )

    parsed = parse_dpo_judge_response(raw, "v4", False)

    assert parsed["quality_score"] == 60.0
    assert parsed["hit"] == 0


def test_transport_or_parse_failure_is_judge_error_not_wrong_answer():
    class FailingClient(JudgeClient):
        def _post_messages(self, messages, *, max_tokens=1024):
            raise OSError("transport failed")

    client = FailingClient(JudgeSettings())
    with pytest.raises(OSError):
        client.judge_messages([{"role": "user", "content": "q"}])

    with pytest.raises(ValueError):
        parse_dpo_judge_response('{"correct": null}', "binary", False)
