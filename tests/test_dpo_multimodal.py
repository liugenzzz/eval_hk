from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest
from PIL import Image

import eval_tool.dpo_multimodal as dpo_multimodal
from eval_tool.dpo_input import DpoCandidate, DpoTurn, ImageRef, SourceRef
from eval_tool.dpo_multimodal import (
    DpoMultimodalError,
    image_data_url,
    inspect_image,
    interleave_content,
    open_model_messages,
    preflight_candidate_images,
)


def _write_image(
    path: Path,
    *,
    format_: str = "PNG",
    color: tuple[int, int, int] = (12, 34, 56),
) -> bytes:
    Image.new("RGB", (2, 2), color).save(path, format=format_)
    return path.read_bytes()


def _source(tmp_path: Path, *, record_index: int = 0) -> SourceRef:
    return SourceRef(
        input_index=0,
        source_path=tmp_path / "input.json",
        container_format="json_object",
        record_index=record_index,
        line_number=None,
        source_id=None,
        raw_digest=f"{record_index:064x}",
    )


def _candidate(
    tmp_path: Path,
    *,
    sample_id: str = "sample-1",
    conversations: tuple[DpoTurn, ...] = (DpoTurn("human", "question"),),
    images: tuple[ImageRef, ...] = (),
    chosen: str = "gold answer",
    record_index: int = 0,
) -> DpoCandidate:
    return DpoCandidate(
        sample_id=sample_id,
        conversations=conversations,
        chosen=chosen,
        images=images,
        source=_source(tmp_path, record_index=record_index),
        source_format="sharegpt",
        turn_index=1,
        turn_count=2,
    )


@pytest.mark.parametrize(
    ("text", "images", "expected"),
    [
        (
            "<image>leading\n",
            ["I1"],
            [
                {"type": "image", "image": "I1"},
                {"type": "text", "text": "leading\n"},
            ],
        ),
        (
            "left\n<image>right",
            ["I1"],
            [
                {"type": "text", "text": "left\n"},
                {"type": "image", "image": "I1"},
                {"type": "text", "text": "right"},
            ],
        ),
        (
            "trailing<image>",
            ["I1"],
            [
                {"type": "text", "text": "trailing"},
                {"type": "image", "image": "I1"},
            ],
        ),
        (
            "<image>A<image><image>B",
            ["I1", "I2", "I3"],
            [
                {"type": "image", "image": "I1"},
                {"type": "text", "text": "A"},
                {"type": "image", "image": "I2"},
                {"type": "image", "image": "I3"},
                {"type": "text", "text": "B"},
            ],
        ),
        (
            "literal <Image> marker",
            [],
            [{"type": "text", "text": "literal <Image> marker"}],
        ),
    ],
)
def test_interleave_content_replaces_only_literal_markers_in_place(
    text, images, expected
):
    assert interleave_content(text, iter(images)) == expected


@pytest.mark.parametrize(
    ("text", "images"),
    [("<image><image>", ["I1"]), ("<image>", ["I1", "I2"]), ("plain", ["I1"])],
)
def test_interleave_content_rejects_too_few_or_too_many_images(text, images):
    with pytest.raises(DpoMultimodalError, match="image"):
        interleave_content(text, iter(images))


def test_inspect_image_detects_png_and_jpeg_mime_from_misleading_extensions(
    tmp_path,
):
    png_path = tmp_path / "actually-png.jpg"
    jpeg_path = tmp_path / "actually-jpeg.png"
    png_bytes = _write_image(png_path, format_="PNG")
    jpeg_bytes = _write_image(jpeg_path, format_="JPEG")

    png = inspect_image(ImageRef(original="actually-png.jpg", resolved=png_path))
    jpeg = inspect_image(ImageRef(original="actually-jpeg.png", resolved=jpeg_path))

    assert png.mime_type == "image/png"
    assert png.byte_count == len(png_bytes)
    assert png.sha256 == hashlib.sha256(png_bytes).hexdigest()
    assert jpeg.mime_type == "image/jpeg"
    assert jpeg.byte_count == len(jpeg_bytes)
    assert jpeg.sha256 == hashlib.sha256(jpeg_bytes).hexdigest()


def test_preflight_deduplicates_resolved_paths_and_preserves_original_refs(
    tmp_path, monkeypatch
):
    path = tmp_path / "image.bin"
    _write_image(path)
    first_ref = ImageRef(original="first/name.png", resolved=path)
    alias_ref = ImageRef(original="different/name.jpeg", resolved=path)
    first = _candidate(
        tmp_path,
        sample_id="first",
        conversations=(DpoTurn("human", "<image>first"),),
        images=(first_ref,),
    )
    alias = _candidate(
        tmp_path,
        sample_id="alias",
        conversations=(DpoTurn("human", "<image>alias"),),
        images=(alias_ref,),
        record_index=1,
    )
    calls: list[ImageRef] = []
    real_inspect = dpo_multimodal.inspect_image

    def counting_inspect(ref):
        calls.append(ref)
        return real_inspect(ref)

    monkeypatch.setattr(dpo_multimodal, "inspect_image", counting_inspect)

    result = preflight_candidate_images([first, alias])

    assert len(calls) == 1
    assert list(result.assets) == [path.resolve()]
    assert result.assets[path.resolve()].ref.original == "first/name.png"
    assert result.issues_by_sample_id == {}
    assert first.images[0].original == "first/name.png"
    assert alias.images[0].original == "different/name.jpeg"


def test_preflight_invalid_images_reject_only_referencing_candidates(tmp_path):
    valid_path = tmp_path / "valid.png"
    empty_path = tmp_path / "empty.png"
    corrupt_path = tmp_path / "corrupt.jpg"
    unsupported_path = tmp_path / "vector.svg"
    missing_path = tmp_path / "missing.png"
    _write_image(valid_path)
    empty_path.write_bytes(b"")
    corrupt_path.write_bytes(b"not a decodable image")
    unsupported_path.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")

    def candidate(sample_id, paths, record_index):
        refs = tuple(
            ImageRef(original=path.name, resolved=path) for path in paths
        )
        markers = "".join("<image>" for _ in refs)
        return _candidate(
            tmp_path,
            sample_id=sample_id,
            conversations=(DpoTurn("human", markers + sample_id),),
            images=refs,
            record_index=record_index,
        )

    candidates = [
        candidate("valid", [valid_path], 0),
        candidate("empty", [empty_path], 1),
        candidate("corrupt", [corrupt_path], 2),
        candidate("unsupported", [unsupported_path], 3),
        candidate("missing", [missing_path], 4),
        candidate("two-bad", [empty_path, corrupt_path], 5),
        candidate("no-images", [], 6),
    ]

    result = preflight_candidate_images(candidates)

    assert list(result.assets) == [valid_path.resolve()]
    assert set(result.issues_by_sample_id) == {
        "empty",
        "corrupt",
        "unsupported",
        "missing",
        "two-bad",
    }
    assert all(
        issue.reason_code == "missing_image"
        for issue in result.issues_by_sample_id.values()
    )
    assert all(
        "unreadable_image" in issue.detail
        for issue in result.issues_by_sample_id.values()
    )
    assert result.issues_by_sample_id["two-bad"].sample_id == "two-bad"


def test_open_model_messages_preserves_turns_markers_order_and_lifetimes(
    tmp_path, monkeypatch
):
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    _write_image(first_path, color=(255, 0, 0))
    _write_image(second_path, color=(0, 255, 0))
    candidate = _candidate(
        tmp_path,
        conversations=(
            DpoTurn("human", "<image>first\n"),
            DpoTurn("gpt", "gold history keeps literal <image>"),
            DpoTurn("human", "before<image>after"),
        ),
        images=(
            ImageRef(original="first-original.png", resolved=first_path),
            ImageRef(original="second-original.png", resolved=second_path),
        ),
        chosen="CHOSEN_MUST_NOT_APPEAR",
    )
    preflight = preflight_candidate_images([candidate])
    source_images = []
    real_open = Image.open

    def tracking_open(*args, **kwargs):
        opened = real_open(*args, **kwargs)
        source_images.append(opened)
        return opened

    monkeypatch.setattr(dpo_multimodal.Image, "open", tracking_open)

    with open_model_messages(candidate, preflight.assets) as messages:
        assert [message["role"] for message in messages] == [
            "user",
            "assistant",
            "user",
        ]
        assert [part["type"] for part in messages[0]["content"]] == [
            "image",
            "text",
        ]
        assert messages[0]["content"][1] == {
            "type": "text",
            "text": "first\n",
        }
        assert messages[1]["content"] == [
            {"type": "text", "text": "gold history keeps literal <image>"}
        ]
        assert [part["type"] for part in messages[2]["content"]] == [
            "text",
            "image",
            "text",
        ]
        assert messages[2]["content"][0]["text"] == "before"
        assert messages[2]["content"][2]["text"] == "after"
        model_images = [
            part["image"]
            for message in messages
            for part in message["content"]
            if part["type"] == "image"
        ]
        assert [image.mode for image in model_images] == ["RGB", "RGB"]
        assert [image.getpixel((0, 0)) for image in model_images] == [
            (255, 0, 0),
            (0, 255, 0),
        ]
        assert all(image.getpixel((0, 0)) for image in source_images)
        all_text = "".join(
            part["text"]
            for message in messages
            for part in message["content"]
            if part["type"] == "text"
        )
        assert "CHOSEN_MUST_NOT_APPEAR" not in all_text

    for image in [*model_images, *source_images]:
        with pytest.raises(ValueError):
            image.getpixel((0, 0))


def test_open_model_messages_closes_images_when_context_body_raises(tmp_path):
    path = tmp_path / "image.png"
    _write_image(path)
    candidate = _candidate(
        tmp_path,
        conversations=(DpoTurn("human", "<image>question"),),
        images=(ImageRef(original="image.png", resolved=path),),
    )
    assets = preflight_candidate_images([candidate]).assets
    model_image = None

    with pytest.raises(RuntimeError, match="consumer failed"):
        with open_model_messages(candidate, assets) as messages:
            model_image = messages[0]["content"][0]["image"]
            assert model_image.getpixel((0, 0)) == (12, 34, 56)
            raise RuntimeError("consumer failed")

    assert model_image is not None
    with pytest.raises(ValueError):
        model_image.getpixel((0, 0))


def test_preflight_fingerprint_change_blocks_model_messages_and_data_url(tmp_path):
    path = tmp_path / "image.png"
    original = _write_image(path)
    candidate = _candidate(
        tmp_path,
        conversations=(DpoTurn("human", "<image>question"),),
        images=(ImageRef(original="kept/original.png", resolved=path),),
    )
    asset = preflight_candidate_images([candidate]).assets[path.resolve()]
    changed = bytearray(original)
    changed[-1] ^= 1
    path.write_bytes(changed)

    with pytest.raises(DpoMultimodalError) as open_error:
        with open_model_messages(candidate, {path.resolve(): asset}):
            pass
    with pytest.raises(DpoMultimodalError) as url_error:
        image_data_url(asset)

    assert "base64" not in str(open_error.value).lower()
    assert "base64" not in str(url_error.value).lower()


def test_image_data_url_uses_verified_original_bytes_and_content_mime(tmp_path):
    path = tmp_path / "misleading.jpeg"
    raw = _write_image(path, format_="PNG")
    asset = inspect_image(ImageRef(original="misleading.jpeg", resolved=path))

    data_url = image_data_url(asset)

    prefix, encoded = data_url.split(",", 1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded) == raw

