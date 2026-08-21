from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .imaging import IMAGE_PREPROCESS_PROFILE, validate_image_pixel_bounds


class InferCacheError(RuntimeError):
    """Inference resume state is malformed, conflicting, or unsafe to reuse."""


_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class InferManifest:
    fingerprint: str
    dataset_name: str
    row_count: int
    index_digest: str


def build_index_digest(indexes: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for index in indexes:
        encoded = str(index).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big"))
        digest.update(encoded)
    return digest.hexdigest()


def write_manifest(path: str | Path, manifest: InferManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(asdict(manifest), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_manifest(path: str | Path) -> InferManifest | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(raw, dict)
            or not isinstance(raw.get("fingerprint"), str)
            or not raw["fingerprint"]
            or not isinstance(raw.get("dataset_name"), str)
            or not raw["dataset_name"]
            or type(raw.get("row_count")) is not int
            or not isinstance(raw.get("index_digest"), str)
            or not raw["index_digest"]
        ):
            raise TypeError("manifest fields have invalid types")
        manifest = InferManifest(
            fingerprint=raw["fingerprint"],
            dataset_name=raw["dataset_name"],
            row_count=raw["row_count"],
            index_digest=raw["index_digest"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InferCacheError(f"invalid inference manifest: {path}") from exc
    if manifest.row_count < 0:
        raise InferCacheError(f"invalid inference manifest row_count: {manifest.row_count}")
    return manifest


class InferenceLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._acquired = False

    def __enter__(self) -> "InferenceLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                str(self.path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError as exc:
            raise InferCacheError(f"inference already running: {self.path}") from exc
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(f"pid={os.getpid()}\n")
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        self._acquired = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)
            self._acquired = False


class InferShardStore:
    def __init__(self, partial_dir: str | Path, dataset_key: str, fingerprint: str):
        self.partial_dir = Path(partial_dir)
        self.dataset_key = str(dataset_key)
        self.fingerprint = str(fingerprint)
        if not _SAFE_TOKEN.fullmatch(self.dataset_key):
            raise InferCacheError(f"unsafe dataset key: {self.dataset_key!r}")
        if not _SAFE_TOKEN.fullmatch(self.fingerprint):
            raise InferCacheError(f"unsafe fingerprint: {self.fingerprint!r}")
        self.warnings: list[str] = []

    @property
    def stem(self) -> str:
        return f"{self.dataset_key}__{self.fingerprint}"

    def shard_path(self, worker_id: int | None = None) -> Path:
        if worker_id is None:
            suffix = ""
        elif type(worker_id) is int and worker_id >= 0:
            suffix = f"__w{worker_id}"
        else:
            raise InferCacheError(f"invalid worker id: {worker_id}")
        return self.partial_dir / f"{self.stem}{suffix}.jsonl"

    def _current_shards(self) -> list[Path]:
        paths = [self.shard_path()]
        paths.extend(sorted(self.partial_dir.glob(f"{self.stem}__w*.jsonl")))
        return [path for path in paths if path.is_file()]

    def _dataset_shards(self) -> list[Path]:
        if not self.partial_dir.is_dir():
            return []
        pattern = re.compile(
            rf"^{re.escape(self.dataset_key)}__(?P<fingerprint>[A-Za-z0-9_.-]+?)"
            rf"(?:__w\d+)?\.jsonl$"
        )
        return [
            path
            for path in sorted(self.partial_dir.glob(f"{self.dataset_key}__*.jsonl"))
            if path.is_file() and pattern.fullmatch(path.name)
        ]

    def _fingerprint_from_path(self, path: Path) -> str:
        prefix = f"{self.dataset_key}__"
        remainder = path.name[len(prefix) : -len(".jsonl")]
        return remainder.rsplit("__w", 1)[0]

    def foreign_fingerprints(self) -> set[str]:
        return {
            fingerprint
            for path in self._dataset_shards()
            if (fingerprint := self._fingerprint_from_path(path)) != self.fingerprint
        }

    def _remove(self, paths: Iterable[Path]) -> None:
        expected_parent = self.partial_dir.resolve()
        for path in paths:
            if path.parent.resolve() != expected_parent:
                raise InferCacheError(f"refusing to remove shard outside {self.partial_dir}: {path}")
            path.unlink(missing_ok=True)

    def clear_current(self) -> None:
        self._remove(self._current_shards())

    def clean_dataset(self) -> None:
        self._remove(self._dataset_shards())

    def append_batch(
        self,
        records: Iterable[tuple[str, str]],
        worker_id: int | None = None,
    ) -> None:
        batch = [(str(index), str(prediction)) for index, prediction in records]
        if not batch:
            return
        seen: set[str] = set()
        for index, _ in batch:
            if not index:
                raise InferCacheError("inference shard records require a non-empty index")
            if index in seen:
                raise InferCacheError(f"duplicate index '{index}' in batch")
            seen.add(index)
        path = self.shard_path(worker_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            for index, prediction in batch:
                stream.write(
                    json.dumps(
                        {"index": index, "prediction": prediction},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            stream.flush()
            os.fsync(stream.fileno())

    def _remove_truncated_tail(self, path: Path, lines: list[str], valid_line_count: int) -> None:
        repaired = "\n".join(lines[:valid_line_count])
        if repaired:
            repaired += "\n"
        temporary = path.with_name(f"{path.name}.repair.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(repaired)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def load(self) -> dict[str, str]:
        loaded: dict[str, str] = {}
        self.warnings = []
        for path in self._current_shards():
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            for line_no, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    is_truncated_tail = line_no == len(lines) and not text.endswith(
                        ("\n", "\r")
                    )
                    if is_truncated_tail:
                        self._remove_truncated_tail(path, lines, line_no - 1)
                        self.warnings.append(f"{path}:{line_no}: truncated JSON removed")
                        continue
                    raise InferCacheError(f"{path}:{line_no}: invalid JSON") from exc
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("index"), str)
                    or not record["index"]
                    or not isinstance(record.get("prediction"), str)
                ):
                    raise InferCacheError(
                        f"{path}:{line_no}: record requires string index and prediction"
                    )
                index = record["index"]
                if index in loaded:
                    raise InferCacheError(f"duplicate index '{index}' across inference shards")
                loaded[index] = record["prediction"]
        return loaded


def _stable_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda pair: str(pair[0]))
        return {str(key): _stable_value(item) for key, item in items}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _stable_value(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)


def build_infer_fingerprint(
    model_path: str | Path,
    prompt_text: str,
    max_new_tokens: int,
    dataset_key: str,
    dataset_name: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
) -> str:
    image_min_pixels, image_max_pixels = validate_image_pixel_bounds(
        image_min_pixels,
        image_max_pixels,
    )
    input_digest = hashlib.sha256()
    for row in rows:
        encoded = json.dumps(
            _stable_value(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        input_digest.update(encoded)
        input_digest.update(b"\n")
    payload = {
        "model_path": str(Path(model_path).resolve()),
        "prompt_text": str(prompt_text),
        "max_new_tokens": int(max_new_tokens),
        "dataset_key": str(dataset_key),
        "dataset_name": str(dataset_name),
        "input_digest": input_digest.hexdigest(),
    }
    if image_min_pixels is not None:
        payload["image_preprocess"] = {
            "profile": IMAGE_PREPROCESS_PROFILE,
            "min_pixels": image_min_pixels,
            "max_pixels": image_max_pixels,
        }
    encoded_payload = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()[:16]
