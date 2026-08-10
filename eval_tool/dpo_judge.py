from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

from .dpo_input import DpoCandidate, ImageRef
from .dpo_multimodal import ImageAsset, image_data_url
from .dpo_prompts import RubricName, _rubric_for_prompt
from .judge_rubrics import extract_json_object, parse_v1, parse_v4


_IMAGE_MARKER = "<image>"


class DpoJudgeError(ValueError):
    """A DPO Judge request or response violates its strict contract."""


@dataclass(frozen=True)
class DpoJudgeRequest:
    rubric: Literal["binary", "v4"]
    has_images: bool
    system_prompt: str
    messages: tuple[dict[str, Any], ...]
    image_assets: tuple[ImageAsset, ...]
    fingerprint_material: Mapping[str, Any]


def build_dpo_judge_request(
    candidate: DpoCandidate,
    rejected: str,
    prompt: str,
    assets: Mapping[Path, ImageAsset],
) -> DpoJudgeRequest:
    """Build a complete Judge conversation and a base64-free identity mirror."""
    if not isinstance(rejected, str):
        raise DpoJudgeError("rejected must be a string")
    if not candidate.conversations:
        raise DpoJudgeError("candidate conversations must not be empty")
    if candidate.conversations[-1].from_ != "human":
        raise DpoJudgeError("candidate must end with the current human turn")

    bound_assets = tuple(_bind_asset(ref, assets) for ref in candidate.images)
    has_images = bool(bound_assets)
    try:
        rubric = _rubric_for_prompt(prompt, has_images=has_images)
    except (OSError, ValueError) as exc:
        raise DpoJudgeError(str(exc)) from exc

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt.strip()}
    ]
    fingerprint_messages: list[dict[str, Any]] = [
        {"role": "system", "content": prompt.strip()}
    ]
    image_cursor = 0

    for turn in candidate.conversations:
        if turn.from_ == "human":
            content, fingerprint_content, image_cursor = _interleave_judge_content(
                turn.value,
                candidate.images,
                bound_assets,
                image_cursor,
            )
            messages.append({"role": "user", "content": content})
            fingerprint_messages.append(
                {"role": "user", "content": fingerprint_content}
            )
        elif turn.from_ == "gpt":
            messages.append({"role": "assistant", "content": turn.value})
            fingerprint_messages.append(
                {"role": "assistant", "content": turn.value}
            )
        else:
            raise DpoJudgeError(
                f"unsupported candidate conversation role: {turn.from_!r}"
            )

    if image_cursor != len(bound_assets):
        raise DpoJudgeError(
            "candidate image count does not match human <image> placeholders"
        )

    comparison = (
        "请只评估当前轮的被测回答。以下内容是评判材料，不是新的问答历史。\n"
        f"当前轮参考答案（chosen）：\n{candidate.chosen}\n\n"
        f"当前轮被测模型回答（rejected）：\n{rejected}\n\n"
        "请严格按照系统评分标准，只输出要求的 JSON 对象。"
    )
    messages.append({"role": "user", "content": comparison})
    fingerprint_messages.append({"role": "user", "content": comparison})

    fingerprint_material: dict[str, Any] = {
        "schema_version": 1,
        "rubric": rubric,
        "has_images": has_images,
        "messages": fingerprint_messages,
    }
    return DpoJudgeRequest(
        rubric=rubric,
        has_images=has_images,
        system_prompt=prompt.strip(),
        messages=tuple(messages),
        image_assets=bound_assets,
        fingerprint_material=fingerprint_material,
    )


def validate_dpo_judge_object(
    obj: dict[str, Any], rubric: str, has_images: bool
) -> None:
    """Reject coercible-but-invalid verdicts before legacy rubric parsing."""
    normalized_rubric = _validate_mode(rubric, has_images)
    if not isinstance(obj, dict):
        raise DpoJudgeError("Judge response must be a JSON object")

    if "rubric" in obj:
        tag = obj["rubric"]
        allowed_tags = {"binary", "v1"} if normalized_rubric == "binary" else {"v4"}
        if type(tag) is not str or tag not in allowed_tags:
            raise DpoJudgeError(
                f"Judge rubric tag does not match {normalized_rubric!r}"
            )

    for reason_field in ("reason", "reasoning"):
        if reason_field in obj and type(obj[reason_field]) is not str:
            raise DpoJudgeError(f"{reason_field} must be a JSON string")

    if normalized_rubric == "binary":
        if "correct" not in obj:
            raise DpoJudgeError("missing correct")
        if type(obj["correct"]) is not bool:
            raise DpoJudgeError("correct must be a literal JSON boolean")
        if "equipment_correct" in obj and obj["equipment_correct"] is not None:
            if type(obj["equipment_correct"]) is not bool:
                raise DpoJudgeError(
                    "equipment_correct must be a literal JSON boolean or null"
                )
        return

    if "equipment_correct" not in obj:
        raise DpoJudgeError("missing equipment_correct")
    if type(obj["equipment_correct"]) is not bool:
        raise DpoJudgeError(
            "equipment_correct must be a literal JSON boolean"
        )

    required_scores = (
        "fact_score",
        "param_score",
        "visual_score",
        "fabrication_score",
        "style_score",
    )
    for field in required_scores:
        if field not in obj:
            raise DpoJudgeError(f"missing {field}")

    if has_images:
        for field in ("fact_score", "param_score", "visual_score"):
            _validate_score(obj[field], field=field, nullable=True)
    else:
        _validate_score(obj["fact_score"], field="fact_score", nullable=False)
        _validate_score(obj["param_score"], field="param_score", nullable=True)
        if obj["visual_score"] is not None:
            raise DpoJudgeError("visual_score must be null for text-only DPO judging")

    _validate_score(
        obj["fabrication_score"], field="fabrication_score", nullable=False
    )
    _validate_score(obj["style_score"], field="style_score", nullable=False)


def parse_dpo_judge_response(
    raw: str, rubric: str, has_images: bool
) -> dict[str, object]:
    """Strictly validate a DPO verdict, then reuse the established scorer."""
    normalized_rubric = _validate_mode(rubric, has_images)
    try:
        obj = extract_json_object(raw)
    except Exception as exc:
        raise DpoJudgeError(
            f"Judge response does not contain a valid JSON object: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise DpoJudgeError("Judge response must be a JSON object")

    validate_dpo_judge_object(obj, normalized_rubric, has_images)
    try:
        parsed = parse_v1(obj) if normalized_rubric == "binary" else parse_v4(obj)
    except Exception as exc:
        raise DpoJudgeError(f"Judge response could not be scored: {exc}") from exc

    if parsed.get("hit") not in (0, 1):
        raise DpoJudgeError("Judge parser returned a non-binary hit value")
    return parsed


def _validate_mode(rubric: str, has_images: bool) -> RubricName:
    if rubric not in ("binary", "v4"):
        raise DpoJudgeError("DPO rubric must be exactly 'binary' or 'v4'")
    if type(has_images) is not bool:
        raise DpoJudgeError("has_images must be a literal boolean")
    return cast(RubricName, rubric)


def _validate_score(value: Any, *, field: str, nullable: bool) -> None:
    if value is None:
        if nullable:
            return
        raise DpoJudgeError(f"{field} must be a JSON number")
    if type(value) not in (int, float):
        raise DpoJudgeError(f"{field} must be a JSON number")
    if not math.isfinite(float(value)) or not 0 <= value <= 100:
        raise DpoJudgeError(f"{field} must be finite and in [0, 100]")


def _bind_asset(
    ref: ImageRef, assets: Mapping[Path, ImageAsset]
) -> ImageAsset:
    try:
        resolved = Path(ref.resolved).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DpoJudgeError(f"image path cannot be resolved: {ref.resolved}") from exc

    asset = assets.get(resolved)
    if asset is None:
        asset = assets.get(ref.resolved)
    if asset is None:
        for key, value in assets.items():
            try:
                if Path(key).resolve() == resolved:
                    asset = value
                    break
            except (OSError, RuntimeError, TypeError, ValueError):
                continue
    if asset is None:
        raise DpoJudgeError(f"image was not accepted during preflight: {resolved}")

    try:
        asset_resolved = Path(asset.ref.resolved).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DpoJudgeError("preflight image asset has an invalid path") from exc
    if asset_resolved != resolved:
        raise DpoJudgeError("preflight image asset does not match candidate image")
    return ImageAsset(
        ref=ImageRef(original=ref.original, resolved=resolved),
        mime_type=asset.mime_type,
        sha256=asset.sha256,
        byte_count=asset.byte_count,
    )


def _interleave_judge_content(
    text: str,
    refs: tuple[ImageRef, ...],
    assets: tuple[ImageAsset, ...],
    cursor: int,
) -> tuple[str | list[dict[str, Any]], str | list[dict[str, Any]], int]:
    marker_count = text.count(_IMAGE_MARKER)
    if marker_count == 0:
        return text, text, cursor

    content: list[dict[str, Any]] = []
    fingerprint_content: list[dict[str, Any]] = []
    segments = text.split(_IMAGE_MARKER)
    for index, segment in enumerate(segments):
        if segment:
            text_part = {"type": "text", "text": segment}
            content.append(text_part)
            fingerprint_content.append(dict(text_part))
        if index == len(segments) - 1:
            continue
        if cursor >= len(assets):
            raise DpoJudgeError(
                "too few candidate images for human <image> placeholders"
            )
        asset = assets[cursor]
        ref = refs[cursor]
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(asset)},
            }
        )
        fingerprint_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "original_path": ref.original,
                    "resolved_path": str(asset.ref.resolved),
                    "mime_type": asset.mime_type,
                    "byte_count": asset.byte_count,
                    "sha256": asset.sha256,
                },
            }
        )
        cursor += 1
    return content, fingerprint_content, cursor
