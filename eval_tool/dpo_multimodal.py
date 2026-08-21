from __future__ import annotations

import base64
import hashlib
import io
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence

from PIL import Image

from .dpo_input import DpoCandidate, ImageRef, InputIssue
from .imaging import prepare_llamafactory_qwen_image


_IMAGE_MARKER = "<image>"


@dataclass(frozen=True)
class ImageAsset:
    ref: ImageRef
    mime_type: str
    sha256: str
    byte_count: int


@dataclass(frozen=True)
class ImagePreflightResult:
    assets: Mapping[Path, ImageAsset]
    issues_by_sample_id: Mapping[str, InputIssue]


class DpoMultimodalError(RuntimeError):
    """A DPO image cannot be used without violating preflight guarantees."""


def inspect_image(ref: ImageRef) -> ImageAsset:
    """Read and verify one image, deriving its MIME type from decoded content."""
    raw = _read_image_bytes(ref.resolved)
    if not raw:
        raise DpoMultimodalError(f"image is empty: {ref.resolved}")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            format_name = source.format
            source.verify()
    except Exception as exc:
        raise DpoMultimodalError(
            f"image content is unreadable or unsupported: {ref.resolved}"
        ) from exc

    mime_type = Image.MIME.get(str(format_name).upper()) if format_name else None
    if not mime_type or not mime_type.startswith("image/") or len(mime_type) <= 6:
        raise DpoMultimodalError(
            f"image format has no reliable image MIME type: {ref.resolved}"
        )

    return ImageAsset(
        ref=ref,
        mime_type=mime_type,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
    )


def preflight_candidate_images(
    candidates: Sequence[DpoCandidate],
) -> ImagePreflightResult:
    """Inspect each resolved path once and reject only candidates that reference failures."""
    unique_refs: dict[Path, ImageRef] = {}
    candidate_paths: list[tuple[DpoCandidate, tuple[Path, ...], tuple[str, ...]]] = []

    for candidate in candidates:
        paths: list[Path] = []
        resolution_failures: list[str] = []
        for ref in candidate.images:
            try:
                path = _resolved_key(ref.resolved)
            except DpoMultimodalError as exc:
                resolution_failures.append(str(exc))
                continue
            paths.append(path)
            unique_refs.setdefault(
                path, ImageRef(original=ref.original, resolved=path)
            )
        candidate_paths.append(
            (candidate, tuple(paths), tuple(resolution_failures))
        )

    assets: dict[Path, ImageAsset] = {}
    failures: dict[Path, str] = {}
    for path, ref in unique_refs.items():
        try:
            assets[path] = inspect_image(ref)
        except Exception as exc:
            failures[path] = str(exc) or type(exc).__name__

    issues: dict[str, InputIssue] = {}
    for candidate, paths, resolution_failures in candidate_paths:
        details = [*resolution_failures]
        details.extend(f"{path}: {failures[path]}" for path in paths if path in failures)
        if not details:
            continue
        issues[candidate.sample_id] = InputIssue(
            source=candidate.source,
            sample_id=candidate.sample_id,
            reason_code="missing_image",
            detail="unreadable_image: " + "; ".join(details),
        )

    return ImagePreflightResult(
        assets=MappingProxyType(assets),
        issues_by_sample_id=MappingProxyType(issues),
    )


def interleave_content(text: str, images: Iterator[Any]) -> list[dict[str, Any]]:
    """Replace each literal image marker with the next image while preserving text."""
    segments = text.split(_IMAGE_MARKER)
    content: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if segment:
            content.append({"type": "text", "text": segment})
        if index == len(segments) - 1:
            continue
        try:
            image = next(images)
        except StopIteration as exc:
            raise DpoMultimodalError(
                "too few images for literal <image> markers"
            ) from exc
        content.append({"type": "image", "image": image})

    try:
        next(images)
    except StopIteration:
        pass
    else:
        raise DpoMultimodalError("too many images for literal <image> markers")

    if not content:
        content.append({"type": "text", "text": text})
    return content


@contextmanager
def open_model_messages(
    candidate: DpoCandidate,
    assets: Mapping[Path, ImageAsset],
    *,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield structured model messages whose owned RGB images stay live in context."""
    model_images: list[Image.Image] = []
    with ExitStack() as source_stack:
        try:
            for ref in candidate.images:
                path = _resolved_key(ref.resolved)
                asset = assets.get(path)
                if asset is None:
                    asset = assets.get(ref.resolved)
                if asset is None:
                    raise DpoMultimodalError(
                        f"image was not accepted during preflight: {path}"
                    )
                raw = _verified_asset_bytes(asset, path=path)
                try:
                    source = Image.open(io.BytesIO(raw))
                    source_stack.callback(source.close)
                    source.load()
                    model_image = prepare_llamafactory_qwen_image(
                        source,
                        image_min_pixels=image_min_pixels,
                        image_max_pixels=image_max_pixels,
                    )
                except Exception as exc:
                    raise DpoMultimodalError(
                        f"verified image could not be opened for inference: {path}"
                    ) from exc
                model_images.append(model_image)

            messages: list[dict[str, Any]] = []
            cursor = 0
            for turn in candidate.conversations:
                if turn.from_ == "human":
                    marker_count = turn.value.count(_IMAGE_MARKER)
                    turn_images = iter(
                        model_images[cursor : cursor + marker_count]
                    )
                    content = interleave_content(turn.value, turn_images)
                    cursor += marker_count
                    messages.append({"role": "user", "content": content})
                elif turn.from_ == "gpt":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": turn.value}
                            ],
                        }
                    )
                else:
                    raise DpoMultimodalError(
                        f"unsupported conversation role: {turn.from_}"
                    )

            if cursor != len(model_images):
                raise DpoMultimodalError(
                    "too many candidate images for human <image> markers"
                )
            yield messages
        finally:
            for image in reversed(model_images):
                image.close()


def image_data_url(asset: ImageAsset) -> str:
    """Encode the unchanged, fingerprint-verified source bytes as a data URL."""
    if not asset.mime_type.startswith("image/") or len(asset.mime_type) <= 6:
        raise DpoMultimodalError("image asset has an invalid MIME type")
    raw = _verified_asset_bytes(asset, path=asset.ref.resolved)
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{asset.mime_type};base64,{encoded}"


def _resolved_key(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DpoMultimodalError(f"image path cannot be resolved: {path}") from exc


def _read_image_bytes(path: Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DpoMultimodalError(f"image cannot be read: {path}") from exc


def _verified_asset_bytes(asset: ImageAsset, *, path: Path) -> bytes:
    raw = _read_image_bytes(path)
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if len(raw) != asset.byte_count or actual_sha256 != asset.sha256:
        raise DpoMultimodalError(
            f"image changed after preflight fingerprinting: {path}"
        )
    return raw
