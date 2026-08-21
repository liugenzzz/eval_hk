from __future__ import annotations

from collections.abc import Callable

import pytest
from PIL import Image

from eval_tool.imaging import (
    IMAGE_PREPROCESS_PROFILE,
    prepare_llamafactory_qwen_image,
    validate_image_pixel_bounds,
)


def assert_image_closed(image: Image.Image) -> None:
    with pytest.raises(ValueError, match="closed image"):
        image.getpixel((0, 0))


def test_image_preprocess_profile_identifies_pinned_implementation() -> None:
    assert IMAGE_PREPROCESS_PROFILE == "llamafactory-0.9.5-qwen-static-v1"


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(None, None), (1, 1), (65_536, 589_824)],
)
def test_validate_image_pixel_bounds_accepts_pair_or_neither(
    minimum: int | None,
    maximum: int | None,
) -> None:
    assert validate_image_pixel_bounds(minimum, maximum) == (minimum, maximum)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(None, 1), (1, None)],
)
def test_validate_image_pixel_bounds_rejects_half_configured_pair(
    minimum: int | None,
    maximum: int | None,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_image_pixel_bounds(minimum, maximum)

    message = str(exc_info.value)
    assert "image_min_pixels" in message
    assert "image_max_pixels" in message
    assert "together" in message or "both" in message


@pytest.mark.parametrize("field", ["image_min_pixels", "image_max_pixels"])
@pytest.mark.parametrize("value", [True, False, 1.5, "1"])
def test_validate_image_pixel_bounds_requires_exact_int_type(
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "image_min_pixels": 1,
        "image_max_pixels": 2,
    }
    values[field] = value

    with pytest.raises(ValueError) as exc_info:
        validate_image_pixel_bounds(
            values["image_min_pixels"],  # type: ignore[arg-type]
            values["image_max_pixels"],  # type: ignore[arg-type]
        )

    message = str(exc_info.value)
    assert field in message
    assert "int" in message


@pytest.mark.parametrize("field", ["image_min_pixels", "image_max_pixels"])
@pytest.mark.parametrize("value", [0, -1])
def test_validate_image_pixel_bounds_requires_positive_values(
    field: str,
    value: int,
) -> None:
    values = {
        "image_min_pixels": 1,
        "image_max_pixels": 2,
    }
    values[field] = value

    with pytest.raises(ValueError) as exc_info:
        validate_image_pixel_bounds(
            values["image_min_pixels"],
            values["image_max_pixels"],
        )

    message = str(exc_info.value)
    assert field in message
    assert "> 0" in message


def test_validate_image_pixel_bounds_rejects_minimum_above_maximum() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_image_pixel_bounds(3, 2)

    message = str(exc_info.value)
    assert "image_min_pixels" in message
    assert "image_max_pixels" in message
    assert "<=" in message


@pytest.mark.parametrize(
    ("mode", "color", "expected_pixel"),
    [
        ("RGB", (10, 20, 30), (10, 20, 30)),
        ("RGBA", (10, 20, 30, 40), (10, 20, 30)),
    ],
)
def test_unconfigured_preparation_returns_independently_owned_rgb_image(
    mode: str,
    color: tuple[int, ...],
    expected_pixel: tuple[int, int, int],
) -> None:
    source = Image.new(mode, (40, 30), color)
    try:
        result = prepare_llamafactory_qwen_image(
            source,
            image_min_pixels=None,
            image_max_pixels=None,
        )
        try:
            assert result is not source
            assert result.mode == "RGB"
            assert result.size == source.size
            assert result.getpixel((0, 0)) == expected_pixel

            result.putpixel((0, 0), (255, 255, 255))
            assert source.getpixel((0, 0)) == color
        finally:
            result.close()

        assert_image_closed(result)
        assert source.getpixel((0, 0)) == color
    finally:
        source.close()


def test_in_range_configured_image_is_an_independent_rgb_copy() -> None:
    source = Image.new("RGB", (100, 100), (1, 2, 3))
    try:
        result = prepare_llamafactory_qwen_image(
            source,
            image_min_pixels=1_000,
            image_max_pixels=20_000,
        )
        try:
            assert result is not source
            assert result.mode == "RGB"
            assert result.size == (100, 100)
            result.putpixel((0, 0), (9, 9, 9))
            assert source.getpixel((0, 0)) == (1, 2, 3)
        finally:
            result.close()
    finally:
        source.close()


def test_large_image_is_reduced_using_llamafactory_integer_dimensions() -> None:
    source = Image.new("RGB", (1_600, 800), "red")
    try:
        result = prepare_llamafactory_qwen_image(
            source,
            image_min_pixels=65_536,
            image_max_pixels=589_824,
        )
        try:
            assert result.size == (1_086, 543)
            assert result.mode == "RGB"
        finally:
            result.close()

        assert source.size == (1_600, 800)
        assert source.getpixel((0, 0)) == (255, 0, 0)
    finally:
        source.close()


def test_small_image_is_enlarged_to_the_minimum_area() -> None:
    source = Image.new("RGB", (100, 100), "blue")
    try:
        result = prepare_llamafactory_qwen_image(
            source,
            image_min_pixels=65_536,
            image_max_pixels=589_824,
        )
        try:
            assert result.size == (256, 256)
            assert result.getpixel((0, 0)) == (0, 0, 255)
        finally:
            result.close()
    finally:
        source.close()


def test_equal_pixel_bounds_run_max_then_min_as_distinct_resize_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resize = Image.Image.resize
    calls: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    created: list[Image.Image] = []

    def resize_spy(
        image: Image.Image,
        size: tuple[int, int],
        *args: object,
        **kwargs: object,
    ) -> Image.Image:
        assert not args
        assert not kwargs
        calls.append((image.size, size, image.mode))
        resized = original_resize(image, size)
        created.append(resized)
        return resized

    monkeypatch.setattr(Image.Image, "resize", resize_spy)
    source = Image.new("RGB", (1_600, 800), "green")
    try:
        result = prepare_llamafactory_qwen_image(
            source,
            image_min_pixels=589_824,
            image_max_pixels=589_824,
        )
        try:
            assert calls == [
                ((1_600, 800), (1_086, 543), "RGB"),
                ((1_086, 543), (1_086, 543), "RGB"),
            ]
            assert result is created[1]
            assert_image_closed(created[0])
            assert result.getpixel((0, 0)) == (0, 128, 0)
        finally:
            result.close()
    finally:
        source.close()


def test_short_edge_is_clamped_to_28_after_area_rules() -> None:
    source = Image.new("RGB", (20, 100), "white")
    try:
        result = prepare_llamafactory_qwen_image(
            source,
            image_min_pixels=1,
            image_max_pixels=10_000,
        )
        try:
            assert result.size == (28, 100)
        finally:
            result.close()
    finally:
        source.close()


@pytest.mark.parametrize(
    ("source_size", "expected_size"),
    [((5_628, 28), (5_040, 28)), ((28, 5_628), (28, 5_040))],
)
def test_extreme_aspect_ratio_is_reduced_from_over_200_to_180(
    source_size: tuple[int, int],
    expected_size: tuple[int, int],
) -> None:
    source = Image.new("RGB", source_size, "black")
    try:
        result = prepare_llamafactory_qwen_image(
            source,
            image_min_pixels=1,
            image_max_pixels=200_000,
        )
        try:
            assert result.size == expected_size
        finally:
            result.close()
    finally:
        source.close()


def test_rgba_area_resize_happens_before_rgb_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resize = Image.Image.resize
    original_convert = Image.Image.convert
    events: list[tuple[object, ...]] = []
    resized_images: list[Image.Image] = []

    def resize_spy(
        image: Image.Image,
        size: tuple[int, int],
        *args: object,
        **kwargs: object,
    ) -> Image.Image:
        is_public_stage = not args and not kwargs
        if is_public_stage:
            events.append(("resize", image.mode, image.size, size, args, kwargs))
        resized = original_resize(image, size, *args, **kwargs)
        if is_public_stage:
            resized_images.append(resized)
        return resized

    def convert_spy(
        image: Image.Image,
        mode: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> Image.Image:
        if mode == "RGB":
            events.append(("convert", image.mode, mode))
        return original_convert(image, mode, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resize_spy)
    monkeypatch.setattr(Image.Image, "convert", convert_spy)
    source = Image.new("RGBA", (1_600, 800), (1, 2, 3, 4))
    try:
        result = prepare_llamafactory_qwen_image(
            source,
            image_min_pixels=65_536,
            image_max_pixels=589_824,
        )
        try:
            assert events == [
                ("resize", "RGBA", (1_600, 800), (1_086, 543), (), {}),
                ("convert", "RGBA", "RGB"),
            ]
            assert result.mode == "RGB"
            assert result.size == (1_086, 543)
            assert_image_closed(resized_images[0])
        finally:
            result.close()

        assert source.mode == "RGBA"
        assert source.size == (1_600, 800)
        assert source.getpixel((0, 0)) == (1, 2, 3, 4)
    finally:
        source.close()


def test_rgba_qwen_dimension_rules_run_after_rgb_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resize = Image.Image.resize
    original_convert = Image.Image.convert
    events: list[tuple[object, ...]] = []

    def resize_spy(
        image: Image.Image,
        size: tuple[int, int],
        *args: object,
        **kwargs: object,
    ) -> Image.Image:
        events.append(("resize", image.mode, size))
        return original_resize(image, size, *args, **kwargs)

    def convert_spy(
        image: Image.Image,
        mode: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> Image.Image:
        events.append(("convert", image.mode, mode))
        return original_convert(image, mode, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resize_spy)
    monkeypatch.setattr(Image.Image, "convert", convert_spy)
    source = Image.new("RGBA", (5_628, 20), (1, 2, 3, 4))
    try:
        result = prepare_llamafactory_qwen_image(
            source,
            image_min_pixels=1,
            image_max_pixels=200_000,
        )
        try:
            assert events == [
                ("convert", "RGBA", "RGB"),
                ("resize", "RGB", (5_628, 28)),
                ("resize", "RGB", (5_040, 28)),
            ]
            assert result.size == (5_040, 28)
        finally:
            result.close()
    finally:
        source.close()


def test_owned_intermediate_is_closed_when_a_later_stage_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_convert: Callable[..., Image.Image] = Image.Image.convert
    converted_images: list[Image.Image] = []

    def convert_spy(
        image: Image.Image,
        mode: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> Image.Image:
        converted = original_convert(image, mode, *args, **kwargs)
        converted_images.append(converted)
        return converted

    def failing_resize(
        image: Image.Image,
        size: tuple[int, int],
        *args: object,
        **kwargs: object,
    ) -> Image.Image:
        raise RuntimeError("synthetic resize failure")

    monkeypatch.setattr(Image.Image, "convert", convert_spy)
    monkeypatch.setattr(Image.Image, "resize", failing_resize)
    source = Image.new("RGBA", (20, 100), (1, 2, 3, 4))
    try:
        with pytest.raises(RuntimeError, match="synthetic resize failure"):
            prepare_llamafactory_qwen_image(
                source,
                image_min_pixels=1,
                image_max_pixels=10_000,
            )

        assert len(converted_images) == 1
        assert_image_closed(converted_images[0])
        assert source.getpixel((0, 0)) == (1, 2, 3, 4)
    finally:
        source.close()
