from __future__ import annotations

import json


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
