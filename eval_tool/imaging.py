from __future__ import annotations

import json
from math import sqrt

from PIL import Image


IMAGE_PREPROCESS_PROFILE = "llamafactory-0.9.5-qwen-static-v1"


def validate_image_pixel_bounds(
    image_min_pixels: int | None,
    image_max_pixels: int | None,
) -> tuple[int | None, int | None]:
    """Validate the optional image-area bounds as one paired setting."""
    if (image_min_pixels is None) != (image_max_pixels is None):
        raise ValueError(
            "image_min_pixels and image_max_pixels must be provided together "
            "or both be None"
        )

    if image_min_pixels is None:
        return None, None

    for field_name, value in (
        ("image_min_pixels", image_min_pixels),
        ("image_max_pixels", image_max_pixels),
    ):
        if type(value) is not int:
            raise ValueError(f"{field_name} must be an int (bool is not accepted)")
        if value <= 0:
            raise ValueError(f"{field_name} must be > 0")

    if image_min_pixels > image_max_pixels:
        raise ValueError("image_min_pixels must be <= image_max_pixels")

    return image_min_pixels, image_max_pixels


def prepare_llamafactory_qwen_image(
    source: Image.Image,
    *,
    image_min_pixels: int | None,
    image_max_pixels: int | None,
) -> Image.Image:
    """Return a caller-owned RGB image using LLaMAFactory's Qwen static rules."""
    image_min_pixels, image_max_pixels = validate_image_pixel_bounds(
        image_min_pixels,
        image_max_pixels,
    )

    current = source
    owns_current = False

    def replace_current(replacement: Image.Image) -> None:
        nonlocal current, owns_current
        previous = current
        owned_previous = owns_current
        current = replacement
        owns_current = replacement is not source
        if owned_previous and previous is not replacement:
            previous.close()

    try:
        if image_min_pixels is not None and image_max_pixels is not None:
            area = current.width * current.height
            if area > image_max_pixels:
                factor = sqrt(image_max_pixels / area)
                replace_current(
                    current.resize(
                        (int(current.width * factor), int(current.height * factor))
                    )
                )

            area = current.width * current.height
            if area < image_min_pixels:
                factor = sqrt(image_min_pixels / area)
                replace_current(
                    current.resize(
                        (int(current.width * factor), int(current.height * factor))
                    )
                )

        if current.mode != "RGB":
            replace_current(current.convert("RGB"))

        if image_min_pixels is not None:
            if min(current.width, current.height) < 28:
                replace_current(
                    current.resize((max(current.width, 28), max(current.height, 28)))
                )

            if current.width / current.height > 200:
                replace_current(current.resize((current.height * 180, current.height)))

            if current.height / current.width > 200:
                replace_current(current.resize((current.width, current.width * 180)))

        if current is source:
            replace_current(source.copy())

        result = current
        owns_current = False
        return result
    except BaseException:
        if owns_current:
            current.close()
        raise


def decode_image_cell(value: object) -> list[str]:
    """A TSV "image" cell holds either one base64 string or a JSON list of them (multi-image rows)."""
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return [str(item) for item in parsed if str(item).strip()]
    return [text]


def encode_image_cell(images_b64: list[str]) -> str:
    if len(images_b64) == 1:
        return images_b64[0]
    return json.dumps(images_b64)
