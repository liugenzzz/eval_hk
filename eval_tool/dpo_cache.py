from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path, PurePath
from typing import Any, BinaryIO, Callable, Iterator, Literal, Mapping, Sequence

from .dpo_config import DpoInferConfig, RubricName
from .dpo_input import DpoCandidate, InputSummary
from .dpo_multimodal import ImageAsset
from .imaging import IMAGE_PREPROCESS_PROFILE, validate_image_pixel_bounds
from .judge import JudgeSettings


DPO_CACHE_SCHEMA_VERSION = 1
DPO_NORMALIZER_VERSION = "dpo-normalizer-v1"
DPO_JUDGE_RESPONSE_SCHEMA_VERSION = "dpo-judge-response-v1"
DPO_JUDGE_PARSER_VERSION = "dpo-judge-parser-v1"
DPO_JUDGE_SCORE_VERSION = "dpo-judge-score-v1"

_ATTEMPT_ID = re.compile(r"^[0-9a-f]{32}$")
_WEIGHT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
_CONTENT_SUFFIXES = {
    ".json",
    ".jinja",
    ".model",
    ".py",
    ".tiktoken",
    ".txt",
    ".yaml",
    ".yml",
}
_MAX_CONTENT_IDENTITY_BYTES = 64 * 1024 * 1024
_SENSITIVE_KEYS = {
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "authorization_header",
}


class DpoCacheError(RuntimeError):
    """DPO phase state is malformed, conflicting, or unsafe to resume."""


@dataclass(frozen=True)
class PhaseAttempt:
    phase: str
    schema_version: int
    fingerprint: str
    expected_id_digest: str
    expected_id_count: int
    attempt_id: str
    key_fields: tuple[str, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "schema_version": self.schema_version,
            "fingerprint": self.fingerprint,
            "expected_id_digest": self.expected_id_digest,
            "expected_id_count": self.expected_id_count,
            "attempt_id": self.attempt_id,
            "key_fields": list(self.key_fields),
        }


@dataclass
class _WriterState:
    stream: BinaryIO
    unsynced_records: int = 0


def canonical_sha256(value: Any) -> str:
    """Return a full SHA-256 over a strict, deterministic JSON representation."""

    try:
        canonical = _canonical_value(value)
        encoded = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, TypeError, UnicodeError, ValueError) as exc:
        raise DpoCacheError("value cannot be canonicalized for a DPO fingerprint") from exc
    return hashlib.sha256(encoded).hexdigest()


def _canonical_value(value: Any) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite floats are not canonical JSON")
        return value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, bytes):
        return {
            "byte_count": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("canonical mappings require string keys")
            if key in normalized:
                raise ValueError("duplicate canonical mapping key")
            normalized[key] = _canonical_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        items = [_canonical_value(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
        )
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def build_phase_fp(
    phase: Literal["inference", "judge_raw", "judge_parse"],
    common_identity: Mapping[str, Any],
    expected_keys: Sequence[tuple[str, ...]],
) -> str:
    if phase not in {"inference", "judge_raw", "judge_parse"}:
        raise DpoCacheError(f"unsupported DPO phase: {phase!r}")
    normalized_keys = _normalize_expected_keys(
        expected_keys, key_count=1 if phase == "inference" else 2
    )
    return canonical_sha256(
        {
            "schema_version": DPO_CACHE_SCHEMA_VERSION,
            "phase": phase,
            "common_identity": common_identity,
            "expected_keys": [list(key) for key in normalized_keys],
        }
    )


class StrictJsonlShardStore:
    """Durable, schema-checked JSONL shards selected through an active attempt."""

    PHASE: str | None = None

    def __init__(
        self,
        phase_dir: Path,
        fingerprint: str,
        key_fields: tuple[str, ...],
        expected_keys: Sequence[tuple[str, ...]],
        schema_validator: Callable[[Mapping[str, Any]], None],
        *,
        fsync_every: int = 32,
        overwrite: bool = False,
        phase: str | None = None,
    ) -> None:
        try:
            self.phase_dir = Path(phase_dir).resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise DpoCacheError("DPO phase directory cannot be resolved") from exc
        if not isinstance(fingerprint, str) or not fingerprint:
            raise DpoCacheError("phase fingerprint must be a non-empty string")
        if (
            not isinstance(key_fields, tuple)
            or not key_fields
            or any(not isinstance(field, str) or not field for field in key_fields)
            or len(set(key_fields)) != len(key_fields)
        ):
            raise DpoCacheError("key_fields must be distinct non-empty strings")
        if not callable(schema_validator):
            raise DpoCacheError("schema_validator must be callable")
        if type(fsync_every) is not int or fsync_every < 1:
            raise DpoCacheError("fsync_every must be a positive integer")
        if type(overwrite) is not bool:
            raise DpoCacheError("overwrite must be a boolean")

        self.fingerprint = fingerprint
        self.key_fields = key_fields
        self.expected_keys = _normalize_expected_keys(
            expected_keys, key_count=len(key_fields)
        )
        self._expected_key_set = frozenset(self.expected_keys)
        self.schema_validator = schema_validator
        self.fsync_every = fsync_every
        self.phase = phase or self.PHASE or self.phase_dir.name
        if not isinstance(self.phase, str) or not self.phase:
            raise DpoCacheError("phase name must be a non-empty string")
        self.expected_id_digest = canonical_sha256(
            [list(key) for key in self.expected_keys]
        )
        self.warnings: list[str] = []
        self._writers: dict[int, _WriterState] = {}
        self._known_keys: set[tuple[str, ...]] | None = None

        try:
            self.phase_dir.mkdir(parents=True, exist_ok=True)
            self.attempts_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DpoCacheError("DPO phase directories cannot be created") from exc
        self._attempt = self._prepare_attempt(overwrite=overwrite)

    @property
    def attempts_dir(self) -> Path:
        return self.phase_dir / "attempts"

    @property
    def active_path(self) -> Path:
        return self.phase_dir / "active.json"

    @property
    def attempt_id(self) -> str:
        return self._attempt.attempt_id

    @property
    def attempt_dir(self) -> Path:
        return self.attempts_dir / self.attempt_id

    @property
    def complete_path(self) -> Path:
        return self.attempt_dir / "complete.json"

    def __enter__(self) -> "StrictJsonlShardStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.sync()

    def shard_path(self, worker_id: int = 0) -> Path:
        worker = _validate_worker_id(worker_id)
        return self.attempt_dir / f"worker-{worker:06d}.jsonl"

    def load(self) -> dict[tuple[str, ...], dict[str, Any]]:
        loaded: dict[tuple[str, ...], dict[str, Any]] = {}
        locations: dict[tuple[str, ...], str] = {}
        self.warnings = []

        for path in self._current_shards():
            for line_number, record in self._load_shard(path):
                key = self._record_key(record, path=path, line_number=line_number)
                if key not in self._expected_key_set:
                    raise DpoCacheError(
                        f"{path}:{line_number}: unknown key {key!r} for this phase"
                    )
                previous = locations.get(key)
                if previous is not None:
                    raise DpoCacheError(
                        f"duplicate key {key!r} across DPO shards "
                        f"({previous} and {path}:{line_number})"
                    )
                loaded[key] = record
                locations[key] = f"{path}:{line_number}"

        ordered = {
            key: loaded[key] for key in self.expected_keys if key in loaded
        }
        self._known_keys = set(ordered)
        return ordered

    def pending_keys(self) -> tuple[tuple[str, ...], ...]:
        loaded = self.load()
        return tuple(key for key in self.expected_keys if key not in loaded)

    def pending_sample_ids(self) -> tuple[str, ...]:
        if self.key_fields != ("sample_id",):
            raise DpoCacheError(
                "pending_sample_ids is only available for sample_id-only stores"
            )
        return tuple(key[0] for key in self.pending_keys())

    def append_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        worker_id: int = 0,
    ) -> None:
        worker = _validate_worker_id(worker_id)
        materialized: list[tuple[dict[str, Any], tuple[str, ...]]] = []
        batch_keys: set[tuple[str, ...]] = set()
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise DpoCacheError(f"batch record {index} must be an object")
            copied = dict(record)
            key = self._record_key(copied, path=None, line_number=index + 1)
            if key not in self._expected_key_set:
                raise DpoCacheError(f"batch record {index}: unknown key {key!r}")
            if key in batch_keys:
                raise DpoCacheError(f"duplicate key {key!r} in append batch")
            batch_keys.add(key)
            materialized.append((copied, key))
        if not materialized:
            return

        if self._known_keys is None:
            self._known_keys = set(self.load())
        overlap = batch_keys.intersection(self._known_keys)
        if overlap:
            key = min(overlap)
            raise DpoCacheError(f"duplicate key {key!r} already exists in phase cache")

        path = self.shard_path(worker)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = _AdvisoryLock(path.with_suffix(path.suffix + ".append.lock"))
        with lock:
            state = self._writers.get(worker)
            if state is None or state.stream.closed:
                state = _WriterState(path.open("ab"))
                self._writers[worker] = state
            try:
                for record, key in materialized:
                    payload = _encode_json_line(record)
                    written = state.stream.write(payload)
                    if written != len(payload):
                        raise OSError("short write while appending DPO cache record")
                    state.unsynced_records += 1
                    self._known_keys.add(key)
                    if state.unsynced_records >= self.fsync_every:
                        state.stream.flush()
                        os.fsync(state.stream.fileno())
                        state.unsynced_records = 0
                state.stream.flush()
            except Exception:
                try:
                    state.stream.flush()
                except Exception:
                    pass
                raise

    def sync(self) -> None:
        errors: list[BaseException] = []
        for worker, state in list(self._writers.items()):
            try:
                if not state.stream.closed:
                    state.stream.flush()
                    if state.unsynced_records:
                        os.fsync(state.stream.fileno())
                        state.unsynced_records = 0
            except BaseException as exc:
                errors.append(exc)
            finally:
                try:
                    state.stream.close()
                except BaseException as exc:
                    errors.append(exc)
                self._writers.pop(worker, None)
        if errors:
            raise DpoCacheError("could not sync and close a DPO phase writer") from errors[0]

    def mark_complete(
        self,
        *,
        snapshot: Mapping[tuple[str, ...], Mapping[str, Any]] | None = None,
    ) -> None:
        self.sync()
        loaded = self.load() if snapshot is None else snapshot
        unknown = set(loaded).difference(self._expected_key_set)
        if unknown:
            raise DpoCacheError(
                f"cannot complete DPO phase with {len(unknown)} unknown records"
            )
        missing = [key for key in self.expected_keys if key not in loaded]
        if missing:
            raise DpoCacheError(
                f"cannot complete DPO phase with {len(missing)} missing records"
            )
        for line_number, key in enumerate(self.expected_keys, start=1):
            record = loaded[key]
            if not isinstance(record, Mapping):
                raise DpoCacheError(
                    f"completion snapshot record {line_number} must be an object"
                )
            actual = self._record_key(record, path=None, line_number=line_number)
            if actual != key:
                raise DpoCacheError(
                    f"completion snapshot key {key!r} does not match record {actual!r}"
                )
        payload = {
            **self._attempt.as_json(),
            "record_count": len(self.expected_keys),
            "record_key_digest": canonical_sha256(
                [list(key) for key in self.expected_keys]
            ),
        }
        _atomic_write_json(self.complete_path, payload)

    def is_complete(self) -> bool:
        if not self.complete_path.is_file():
            return False
        raw = _read_json_object(self.complete_path, label="phase complete marker")
        expected = self._attempt.as_json()
        if set(raw) != {*expected, "record_count", "record_key_digest"}:
            raise DpoCacheError(f"invalid phase complete marker: {self.complete_path}")
        for key, value in expected.items():
            if raw.get(key) != value:
                raise DpoCacheError(
                    f"phase complete marker does not match active attempt: {self.complete_path}"
                )
        if type(raw.get("record_count")) is not int:
            raise DpoCacheError(f"invalid phase complete marker: {self.complete_path}")
        loaded = self.load()
        if len(loaded) != len(self.expected_keys):
            raise DpoCacheError("complete DPO phase is missing cached records")
        expected_digest = canonical_sha256([list(key) for key in loaded])
        if raw.get("record_key_digest") != expected_digest:
            raise DpoCacheError("complete DPO phase record digest does not match")
        return True

    def start_new_attempt(self) -> str:
        """Durably switch to a fresh attempt while preserving all older attempts."""

        self.sync()
        self._attempt = self._create_attempt()
        self._known_keys = None
        self.warnings = []
        return self.attempt_id

    def _prepare_attempt(self, *, overwrite: bool) -> PhaseAttempt:
        if self.active_path.is_file():
            active = _phase_attempt_from_json(
                _read_json_object(self.active_path, label="active phase pointer"),
                path=self.active_path,
            )
            self._verify_attempt(active)
            if self._same_identity(active):
                return active
            if not overwrite:
                raise DpoCacheError(
                    "active DPO phase identity differs; pass overwrite to switch attempts"
                )
        return self._create_attempt()

    def _create_attempt(self) -> PhaseAttempt:
        while True:
            attempt_id = uuid.uuid4().hex
            attempt_dir = self.attempts_dir / attempt_id
            try:
                attempt_dir.mkdir()
            except FileExistsError:
                continue
            break
        _fsync_directory(self.attempts_dir)
        attempt = PhaseAttempt(
            phase=self.phase,
            schema_version=DPO_CACHE_SCHEMA_VERSION,
            fingerprint=self.fingerprint,
            expected_id_digest=self.expected_id_digest,
            expected_id_count=len(self.expected_keys),
            attempt_id=attempt_id,
            key_fields=self.key_fields,
        )
        _atomic_write_json(attempt_dir / "attempt.json", attempt.as_json())
        _fsync_directory(attempt_dir)
        _atomic_write_json(self.active_path, attempt.as_json())
        return attempt

    def _verify_attempt(self, attempt: PhaseAttempt) -> None:
        attempt_dir = self.attempts_dir / attempt.attempt_id
        try:
            resolved = attempt_dir.resolve()
            expected_parent = self.attempts_dir.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            raise DpoCacheError("active DPO attempt path cannot be resolved") from exc
        if resolved.parent != expected_parent or not resolved.is_dir():
            raise DpoCacheError("active DPO attempt does not exist inside attempts")
        metadata_path = resolved / "attempt.json"
        metadata = _phase_attempt_from_json(
            _read_json_object(metadata_path, label="phase attempt metadata"),
            path=metadata_path,
        )
        if metadata != attempt:
            raise DpoCacheError("active pointer conflicts with phase attempt metadata")

    def _same_identity(self, attempt: PhaseAttempt) -> bool:
        return (
            attempt.phase == self.phase
            and attempt.schema_version == DPO_CACHE_SCHEMA_VERSION
            and attempt.fingerprint == self.fingerprint
            and attempt.expected_id_digest == self.expected_id_digest
            and attempt.expected_id_count == len(self.expected_keys)
            and attempt.key_fields == self.key_fields
        )

    def _current_shards(self) -> list[Path]:
        if not self.attempt_dir.is_dir():
            raise DpoCacheError("active DPO attempt directory disappeared")
        pattern = re.compile(r"^worker-(\d{6,})\.jsonl$")
        shards: list[tuple[int, Path]] = []
        for path in self.attempt_dir.glob("worker-*.jsonl"):
            match = pattern.fullmatch(path.name)
            if match and path.is_file():
                shards.append((int(match.group(1)), path))
        return [path for _, path in sorted(shards)]

    def _load_shard(self, path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
        line_number = 0
        byte_offset = 0
        tail: bytes | None = None
        tail_offset = 0
        try:
            with path.open("rb") as stream:
                while True:
                    raw_line = stream.readline()
                    if not raw_line:
                        break
                    line_number += 1
                    line_offset = byte_offset
                    byte_offset += len(raw_line)
                    if raw_line.endswith(b"\n"):
                        yield (
                            line_number,
                            self._parse_record_bytes(
                                raw_line[:-1], path, line_number
                            ),
                        )
                        continue
                    tail = raw_line
                    tail_offset = line_offset
                    break
        except OSError as exc:
            raise DpoCacheError(f"cannot read DPO cache shard: {path}") from exc

        if tail is None:
            return
        try:
            text = tail.decode("utf-8")
        except UnicodeDecodeError as exc:
            if exc.reason == "unexpected end of data" and exc.end == len(tail):
                self._truncate_tail(path, tail_offset, line_number)
                return
            raise DpoCacheError(f"{path}:{line_number}: invalid UTF-8") from exc

        try:
            record = _strict_json_object(text)
        except json.JSONDecodeError as exc:
            if _is_truncated_json(text, exc):
                self._truncate_tail(path, tail_offset, line_number)
                return
            raise DpoCacheError(f"{path}:{line_number}: invalid JSON") from exc
        except (TypeError, ValueError) as exc:
            raise DpoCacheError(
                f"{path}:{line_number}: invalid JSON record: {exc}"
            ) from exc

        self._validate_schema(record, path=path, line_number=line_number)
        _append_newline(path)
        yield line_number, record

    def _parse_record_bytes(
        self, raw_line: bytes, path: Path, line_number: int
    ) -> dict[str, Any]:
        if not raw_line.strip():
            raise DpoCacheError(f"{path}:{line_number}: blank JSONL records are invalid")
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DpoCacheError(f"{path}:{line_number}: invalid UTF-8") from exc
        try:
            record = _strict_json_object(text)
        except json.JSONDecodeError as exc:
            raise DpoCacheError(f"{path}:{line_number}: invalid JSON") from exc
        except (TypeError, ValueError) as exc:
            raise DpoCacheError(
                f"{path}:{line_number}: invalid JSON record: {exc}"
            ) from exc
        self._validate_schema(record, path=path, line_number=line_number)
        return record

    def _record_key(
        self,
        record: Mapping[str, Any],
        *,
        path: Path | None,
        line_number: int,
    ) -> tuple[str, ...]:
        location = f"{path}:{line_number}" if path is not None else f"record {line_number}"
        values: list[str] = []
        for field in self.key_fields:
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise DpoCacheError(
                    f"{location}: key field {field!r} must be a non-empty string"
                )
            values.append(value)
        self._validate_schema(record, path=path, line_number=line_number)
        return tuple(values)

    def _validate_schema(
        self, record: Mapping[str, Any], *, path: Path | None, line_number: int
    ) -> None:
        try:
            self.schema_validator(record)
        except DpoCacheError:
            raise
        except Exception as exc:
            location = f"{path}:{line_number}" if path is not None else f"record {line_number}"
            detail = str(exc) or type(exc).__name__
            raise DpoCacheError(f"{location}: invalid schema: {detail}") from exc

    def _truncate_tail(self, path: Path, byte_count: int, line_number: int) -> None:
        try:
            with path.open("r+b") as stream:
                stream.truncate(byte_count)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise DpoCacheError(f"cannot repair truncated DPO shard: {path}") from exc
        self.warnings.append(f"{path}:{line_number}: truncated tail removed")


class DpoInferenceStore(StrictJsonlShardStore):
    PHASE = "inference"

    def __init__(
        self,
        phase_dir: Path,
        fingerprint: str,
        expected_sample_ids: Sequence[str] | Sequence[tuple[str, ...]],
        *,
        fsync_every: int = 32,
        overwrite: bool = False,
    ) -> None:
        super().__init__(
            phase_dir,
            fingerprint,
            ("sample_id",),
            _sample_expected_keys(expected_sample_ids),
            _validate_inference_record,
            fsync_every=fsync_every,
            overwrite=overwrite,
        )


class DpoJudgeRawStore(StrictJsonlShardStore):
    PHASE = "judge_raw"

    def __init__(
        self,
        phase_dir: Path,
        fingerprint: str,
        expected_keys: Sequence[tuple[str, str]],
        *,
        fsync_every: int = 32,
        overwrite: bool = False,
    ) -> None:
        super().__init__(
            phase_dir,
            fingerprint,
            ("sample_id", "judge_request_fp"),
            expected_keys,
            _validate_judge_raw_record,
            fsync_every=fsync_every,
            overwrite=overwrite,
        )


class DpoJudgeParseStore(StrictJsonlShardStore):
    PHASE = "judge_parse"

    def __init__(
        self,
        phase_dir: Path,
        fingerprint: str,
        expected_keys: Sequence[tuple[str, str]],
        *,
        fsync_every: int = 32,
        overwrite: bool = False,
    ) -> None:
        super().__init__(
            phase_dir,
            fingerprint,
            ("sample_id", "judge_parse_fp"),
            expected_keys,
            _validate_judge_parse_record,
            fsync_every=fsync_every,
            overwrite=overwrite,
        )


class _AdvisoryLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: BinaryIO | None = None

    def __enter__(self) -> "_AdvisoryLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        stream = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            if os.fstat(descriptor).st_size == 0:
                stream.write(b"\0")
            stream.seek(0)
            _lock_descriptor(stream, blocking=True)
        except Exception:
            stream.close()
            raise
        self._stream = stream
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            _unlock_descriptor(stream)
        finally:
            stream.close()


class DpoRunLock:
    """Cross-process advisory lock whose OS descriptor, not its text, owns it."""

    def __init__(self, path: Path) -> None:
        try:
            self.path = Path(path).resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise DpoCacheError("DPO run lock path cannot be resolved") from exc
        self.owner_token = uuid.uuid4().hex
        self._owner_pid: int | None = None
        self._stream: BinaryIO | None = None

    @classmethod
    def for_identity(cls, output_dir: Path, work_dir: Path) -> "DpoRunLock":
        return cls(run_lock_path(output_dir, work_dir))

    def __enter__(self) -> "DpoRunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

    def acquire(self) -> "DpoRunLock":
        if self._stream is not None:
            raise DpoCacheError("this DPO run lock is already acquired")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        stream = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            if os.fstat(descriptor).st_size == 0:
                stream.write(b"\0")
            stream.seek(0)
            _lock_descriptor(stream)
        except (OSError, BlockingIOError) as exc:
            stream.close()
            raise DpoCacheError(
                f"DPO build is already running for this identity: {self.path}"
            ) from exc
        except Exception:
            stream.close()
            raise

        try:
            owner = _encode_json_line(
                {
                    "owner_token": self.owner_token,
                    "pid": os.getpid(),
                }
            )
            stream.seek(0)
            stream.write(owner)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
        except Exception:
            try:
                _unlock_descriptor(stream)
            finally:
                stream.close()
            raise
        self._stream = stream
        self._owner_pid = os.getpid()
        return self

    def release(self) -> None:
        if (
            self._stream is None
            or self._owner_pid is None
            or self._owner_pid != os.getpid()
        ):
            raise DpoCacheError("this object does not own the DPO run lock")
        stream = self._stream
        self._stream = None
        self._owner_pid = None
        try:
            _unlock_descriptor(stream)
        except OSError as exc:
            raise DpoCacheError("could not release the DPO run lock") from exc
        finally:
            stream.close()


def run_lock_path(output_dir: Path, work_dir: Path) -> Path:
    try:
        output = str(Path(output_dir).resolve())
        work = Path(work_dir).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DpoCacheError("DPO output/work lock identity cannot be resolved") from exc
    digest = canonical_sha256({"output_dir": output, "work_dir": str(work)})
    return work / "locks" / f"dpo-run-{digest}.lock"


def _lock_descriptor(stream: BinaryIO, *, blocking: bool = False) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
        msvcrt.locking(stream.fileno(), mode, 1)
    else:
        import fcntl

        mode = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        fcntl.flock(stream.fileno(), mode)


def _unlock_descriptor(stream: BinaryIO) -> None:
    stream.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def checkpoint_identity(model_path: Path) -> dict[str, Any]:
    """Fingerprint a checkpoint, using stat identity for multi-gigabyte weights.

    Configuration, tokenizer, processor, chat-template, generation, remote-code,
    and index files are content-hashed. Weight files deliberately use relative
    name, size, and ``mtime_ns`` rather than reading every weight byte.
    """

    try:
        root = Path(model_path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DpoCacheError("model path cannot be resolved for checkpoint identity") from exc
    if not root.exists():
        raise DpoCacheError(f"model path does not exist: {root}")
    if root.is_file():
        entries = [(root.name, root)]
        identity_root = root.parent
    elif root.is_dir():
        try:
            entries = sorted(
                (
                    (path.relative_to(root).as_posix(), path)
                    for path in root.rglob("*")
                    if path.is_file()
                ),
                key=lambda pair: pair[0],
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise DpoCacheError(f"cannot enumerate checkpoint files: {root}") from exc
        identity_root = root
    else:
        raise DpoCacheError(f"model path is not a regular file or directory: {root}")

    files: list[dict[str, Any]] = []
    for relative, path in entries:
        kind = _checkpoint_file_kind(relative, path)
        if kind is None:
            continue
        try:
            stat = path.stat()
        except OSError as exc:
            raise DpoCacheError(f"cannot stat checkpoint file: {path}") from exc
        if kind == "weight_stat":
            files.append(
                {
                    "path": relative,
                    "kind": kind,
                    "byte_count": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
            continue
        if stat.st_size > _MAX_CONTENT_IDENTITY_BYTES:
            files.append(
                {
                    "path": relative,
                    "kind": "large_content_stat",
                    "byte_count": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
            continue
        files.append(
            {
                "path": relative,
                "kind": "content_sha256",
                "byte_count": stat.st_size,
                "sha256": _sha256_file(path),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_path": str(root),
        "identity_root": str(identity_root),
        "weight_identity": "relative_path+byte_count+mtime_ns (stat identity)",
        "files": files,
    }
    payload["digest"] = canonical_sha256(payload)
    return payload


def _checkpoint_file_kind(relative: str, path: Path) -> str | None:
    lowered = relative.casefold()
    suffix = path.suffix.casefold()
    if suffix in _CONTENT_SUFFIXES or lowered.endswith(".index.json"):
        return "content_sha256"
    if suffix in _WEIGHT_SUFFIXES:
        return "weight_stat"
    return None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise DpoCacheError(f"cannot hash checkpoint file: {path}") from exc
    return digest.hexdigest()


def build_dpo_inference_fp(
    input_summaries: Sequence[InputSummary],
    candidates: Sequence[DpoCandidate],
    assets: Mapping[Path, ImageAsset],
    config: DpoInferConfig,
    checkpoint: Mapping[str, Any],
) -> str:
    image_min_pixels, image_max_pixels = validate_image_pixel_bounds(
        config.image_min_pixels,
        config.image_max_pixels,
    )
    candidate_material: list[dict[str, Any]] = []
    for candidate in candidates:
        image_material = [
            _image_asset_identity(_asset_for_candidate_ref(ref.resolved, assets))
            for ref in candidate.images
        ]
        candidate_material.append(
            {
                "sample_id": candidate.sample_id,
                "messages": [
                    {"from": turn.from_, "value": turn.value}
                    for turn in candidate.conversations
                ],
                "images": image_material,
            }
        )

    model_identity = {
        "model_name": config.model_name,
        "model_path": str(config.model_path),
        "enable_thinking": config.enable_thinking,
        "max_new_tokens": config.max_new_tokens,
        "batch_size": config.batch_size,
        "torch_dtype": config.torch_dtype,
        "device_map": config.device_map,
        "gpu_ids": list(config.gpu_ids),
    }
    if image_min_pixels is not None:
        model_identity["image_preprocess"] = {
            "profile": IMAGE_PREPROCESS_PROFILE,
            "min_pixels": image_min_pixels,
            "max_pixels": image_max_pixels,
        }

    common_identity = {
        "normalizer_version": DPO_NORMALIZER_VERSION,
        "inputs": [
            {
                "source_path": str(summary.source_path),
                "sha256": summary.sha256,
                "byte_count": summary.byte_count,
                "container_format": summary.container_format,
                "record_count": summary.record_count,
            }
            for summary in input_summaries
        ],
        "model": model_identity,
        "checkpoint": checkpoint,
        "candidates": candidate_material,
    }
    return build_phase_fp(
        "inference",
        common_identity,
        tuple((candidate.sample_id,) for candidate in candidates),
    )


def build_judge_request_fp(
    inference_fp: str,
    settings: JudgeSettings,
    request_fingerprint_material: Mapping[str, Any],
    assets: Sequence[ImageAsset],
    *,
    max_tokens: int = 1024,
) -> str:
    if type(max_tokens) is not int or max_tokens < 1:
        raise DpoCacheError("Judge max_tokens must be a positive integer")
    if not isinstance(inference_fp, str) or not inference_fp:
        raise DpoCacheError("inference fingerprint must be a non-empty string")
    material = {
        "schema_version": 1,
        "inference_fp": inference_fp,
        "transport": {
            "api_base": settings.api_base,
            "model": settings.model,
            "temperature": settings.temperature,
            "timeout": settings.timeout,
            "max_retries": settings.max_retries,
            "max_tokens": max_tokens,
        },
        "request": _without_sensitive_keys(request_fingerprint_material),
        "assets": [_image_asset_identity(asset) for asset in assets],
    }
    return canonical_sha256(material)


def build_judge_parse_fp(
    request_fp: str,
    raw_response: str,
    rubric: RubricName,
    has_images: bool,
) -> str:
    if not isinstance(request_fp, str) or not request_fp:
        raise DpoCacheError("Judge request fingerprint must be a non-empty string")
    if not isinstance(raw_response, str):
        raise DpoCacheError("Judge raw response must be a string")
    if rubric not in {"binary", "v4"}:
        raise DpoCacheError(f"unsupported DPO rubric: {rubric!r}")
    if type(has_images) is not bool:
        raise DpoCacheError("has_images must be a boolean")
    return canonical_sha256(
        {
            "request_fp": request_fp,
            "raw_response": raw_response,
            "rubric": rubric,
            "has_images": has_images,
            "response_schema_version": DPO_JUDGE_RESPONSE_SCHEMA_VERSION,
            "parser_version": DPO_JUDGE_PARSER_VERSION,
            "score_version": DPO_JUDGE_SCORE_VERSION,
        }
    )


def _image_asset_identity(asset: ImageAsset) -> dict[str, Any]:
    return {
        "original_path": asset.ref.original,
        "resolved_path": str(asset.ref.resolved),
        "mime_type": asset.mime_type,
        "sha256": asset.sha256,
        "byte_count": asset.byte_count,
    }


def _asset_for_candidate_ref(
    path: Path, assets: Mapping[Path, ImageAsset]
) -> ImageAsset:
    asset = assets.get(path)
    if asset is not None:
        return asset
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise DpoCacheError(f"candidate image path cannot be resolved: {path}") from exc
    asset = assets.get(resolved)
    if asset is None:
        raise DpoCacheError(f"candidate image has no preflight asset: {resolved}")
    return asset


def _without_sensitive_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_sensitive_keys(item)
            for key, item in value.items()
            if str(key).casefold() not in _SENSITIVE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_without_sensitive_keys(item) for item in value]
    return value


def _validate_inference_record(record: Mapping[str, Any]) -> None:
    required = {"sample_id", "status", "prediction", "error_type", "error_message"}
    _require_exact_fields(record, required)
    _require_nonempty_string(record["sample_id"], "sample_id")
    status = record["status"]
    if status == "ok":
        if not isinstance(record["prediction"], str):
            raise ValueError("ok inference prediction must be a string")
        if record["error_type"] is not None or record["error_message"] is not None:
            raise ValueError("ok inference errors must be null")
    elif status == "error":
        if record["prediction"] is not None:
            raise ValueError("failed inference prediction must be null")
        _require_nonempty_string(record["error_type"], "error_type")
        _require_nonempty_string(record["error_message"], "error_message")
    else:
        raise ValueError("inference status must be ok or error")


def _validate_judge_raw_record(record: Mapping[str, Any]) -> None:
    _require_exact_fields(
        record, {"sample_id", "judge_request_fp", "raw_response"}
    )
    _require_nonempty_string(record["sample_id"], "sample_id")
    _require_nonempty_string(record["judge_request_fp"], "judge_request_fp")
    if not isinstance(record["raw_response"], str):
        raise ValueError("raw_response must be a string")


def _validate_judge_parse_record(record: Mapping[str, Any]) -> None:
    required = {
        "sample_id",
        "judge_parse_fp",
        "status",
        "result",
        "error_type",
        "error_message",
    }
    _require_exact_fields(record, required)
    _require_nonempty_string(record["sample_id"], "sample_id")
    _require_nonempty_string(record["judge_parse_fp"], "judge_parse_fp")
    status = record["status"]
    if status == "ok":
        if not isinstance(record["result"], Mapping):
            raise ValueError("ok Judge parse result must be an object")
        if record["error_type"] is not None or record["error_message"] is not None:
            raise ValueError("ok Judge parse errors must be null")
    elif status == "error":
        if record["result"] is not None:
            raise ValueError("failed Judge parse result must be null")
        _require_nonempty_string(record["error_type"], "error_type")
        _require_nonempty_string(record["error_message"], "error_message")
    else:
        raise ValueError("Judge parse status must be ok or error")


def _require_exact_fields(record: Mapping[str, Any], expected: set[str]) -> None:
    actual = set(record)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(str(key) for key in actual - expected)
        raise ValueError(f"record fields differ (missing={missing}, extra={extra})")


def _require_nonempty_string(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")


def _sample_expected_keys(
    values: Sequence[str] | Sequence[tuple[str, ...]],
) -> tuple[tuple[str, ...], ...]:
    keys: list[tuple[str, ...]] = []
    for value in values:
        if isinstance(value, str):
            keys.append((value,))
        else:
            keys.append(tuple(value))
    return tuple(keys)


def _normalize_expected_keys(
    expected_keys: Sequence[tuple[str, ...]], *, key_count: int | None
) -> tuple[tuple[str, ...], ...]:
    if isinstance(expected_keys, (str, bytes)):
        raise DpoCacheError("expected_keys must be a sequence of tuples")
    normalized: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for index, key in enumerate(expected_keys):
        if isinstance(key, (str, bytes)):
            raise DpoCacheError(f"expected key {index} must be a tuple")
        try:
            item = tuple(key)
        except TypeError as exc:
            raise DpoCacheError(f"expected key {index} must be iterable") from exc
        if key_count is not None and len(item) != key_count:
            raise DpoCacheError(
                f"expected key {index} must contain {key_count} fields"
            )
        if not item or any(not isinstance(value, str) or not value for value in item):
            raise DpoCacheError(
                f"expected key {index} must contain non-empty strings"
            )
        if item in seen:
            raise DpoCacheError(f"duplicate expected key: {item!r}")
        seen.add(item)
        normalized.append(item)
    return tuple(normalized)


def _validate_worker_id(worker_id: int) -> int:
    if type(worker_id) is not int or worker_id < 0:
        raise DpoCacheError("worker_id must be a non-negative integer")
    return worker_id


def _phase_attempt_from_json(raw: Mapping[str, Any], *, path: Path) -> PhaseAttempt:
    expected_fields = {
        "phase",
        "schema_version",
        "fingerprint",
        "expected_id_digest",
        "expected_id_count",
        "attempt_id",
        "key_fields",
    }
    if set(raw) != expected_fields:
        raise DpoCacheError(f"invalid phase attempt fields: {path}")
    key_fields = raw.get("key_fields")
    if (
        not isinstance(raw.get("phase"), str)
        or not raw["phase"]
        or type(raw.get("schema_version")) is not int
        or not isinstance(raw.get("fingerprint"), str)
        or not raw["fingerprint"]
        or not isinstance(raw.get("expected_id_digest"), str)
        or len(raw["expected_id_digest"]) != 64
        or type(raw.get("expected_id_count")) is not int
        or raw["expected_id_count"] < 0
        or not isinstance(raw.get("attempt_id"), str)
        or _ATTEMPT_ID.fullmatch(raw["attempt_id"]) is None
        or not isinstance(key_fields, list)
        or not key_fields
        or any(not isinstance(field, str) or not field for field in key_fields)
    ):
        raise DpoCacheError(f"invalid phase attempt metadata: {path}")
    return PhaseAttempt(
        phase=raw["phase"],
        schema_version=raw["schema_version"],
        fingerprint=raw["fingerprint"],
        expected_id_digest=raw["expected_id_digest"],
        expected_id_count=raw["expected_id_count"],
        attempt_id=raw["attempt_id"],
        key_fields=tuple(key_fields),
    )


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        value = json.loads(raw, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DpoCacheError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise DpoCacheError(f"invalid {label}: {path}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    payload = _encode_json_line(value)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _encode_json_line(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (OverflowError, TypeError, UnicodeError, ValueError) as exc:
        raise DpoCacheError("DPO cache record is not strict JSON serializable") from exc


def _strict_json_object(text: str) -> dict[str, Any]:
    value = json.loads(
        text,
        object_pairs_hook=_strict_object_pairs,
        parse_constant=_reject_json_constant,
        parse_float=_finite_float,
    )
    if not isinstance(value, dict):
        raise TypeError("JSONL record must be an object")
    return value


def _strict_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _finite_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("non-finite JSON number")
    return result


def _is_truncated_json(text: str, error: json.JSONDecodeError) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if error.msg.startswith("Unterminated string"):
        return True
    if error.pos >= len(stripped):
        return True
    suffix = stripped[error.pos:]
    incomplete_literals = {"t", "tr", "tru", "f", "fa", "fal", "fals", "n", "nu", "nul"}
    if suffix in incomplete_literals:
        return True
    if re.fullmatch(r"-?(?:\d+\.?|\d*\.\d+)(?:[eE][+-]?)?", suffix):
        return suffix.endswith((".", "e", "E", "+", "-"))
    return False


def _append_newline(path: Path) -> None:
    try:
        with path.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise DpoCacheError(f"cannot terminate final DPO JSONL record: {path}") from exc
