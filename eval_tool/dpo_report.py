from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_args

from .dpo_input import ReasonCode


DPO_MANIFEST_SCHEMA_VERSION = 1

_FIXED_ARTIFACT_NAMES = (
    "audit_records.jsonl",
    "rejected_records.jsonl",
    "summary.json",
    "warnings.log",
    "manifest.json",
)
_RESERVED_OUTPUT_NAMES = frozenset(_FIXED_ARTIFACT_NAMES)
_JSONL_FIXED_NAMES = frozenset(
    {"audit_records.jsonl", "rejected_records.jsonl"}
)
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([a-z0-9_-]*(?:api[_-]?key|authorization|access[_-]?token|"
    r"auth[_-]?token|bearer[_-]?token|client[_-]?secret|password|secret))\b"
    r"(\s*(?::|=)\s*)"
    r"(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|[^\s,;]+)",
)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[^\s,;\"']+")
_DATA_URL_RE = re.compile(r"(?i)(?<![A-Za-z0-9+.-])data:[^\s\"']+")
_WINDOWS_DEVICE_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        "clock$",
        "conin$",
        "conout$",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)


class DpoReportError(RuntimeError):
    """A DPO artifact set is invalid or cannot be safely published."""


@dataclass(frozen=True)
class ArtifactStat:
    sha256: str
    byte_count: int
    line_count: int | None


@dataclass(frozen=True)
class AuditRecord:
    source: Mapping[str, Any]
    sample_id: str | None
    source_format: Literal["alpaca", "sharegpt"] | None
    turn_index: int | None
    turn_count: int | None
    modality: Literal["text", "single_image", "multi_image"] | None
    digests: Mapping[str, str]
    prediction: str | None
    inference_error: str | None
    judge: Mapping[str, Any] | None
    disposition: Literal["selected", "rejected"]
    reason_code: ReasonCode | None
    detail: str


def redact_config(value: Any) -> Any:
    """Return a JSON-compatible copy with credentials and data URLs removed."""
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            text_key = str(key)
            if _is_secret_key(text_key):
                redacted[text_key] = "<redacted>"
            else:
                redacted[text_key] = redact_config(item)
        return redacted
    if isinstance(value, (list, tuple)):
        return [redact_config(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [redact_config(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return _redact_text(value)
    if value is None or type(value) in (bool, int, float):
        return value
    return _redact_text(str(value))


def compute_artifact_stat(path: Path, *, jsonl: bool) -> ArtifactStat:
    """Hash an artifact and, for JSONL, strictly validate/count its records."""
    candidate = Path(path)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with candidate.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
        line_count = _validate_jsonl_bytes(candidate.read_bytes()) if jsonl else None
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DpoReportError(f"invalid artifact {candidate}: {exc}") from exc
    return ArtifactStat(
        sha256=digest.hexdigest(),
        byte_count=byte_count,
        line_count=line_count,
    )


def build_summary(audit_records: Sequence[AuditRecord]) -> dict[str, Any]:
    """Build reconciliable audit aggregates using stable bucket names."""
    by_input_file: Counter[str] = Counter()
    by_source_format: Counter[str] = Counter(
        {"alpaca": 0, "sharegpt": 0, "invalid": 0}
    )
    by_modality: Counter[str] = Counter(
        {"text": 0, "single_image": 0, "multi_image": 0, "invalid": 0}
    )
    by_turn_shape: Counter[str] = Counter(
        {"single_turn": 0, "multi_turn": 0, "invalid": 0}
    )
    by_disposition: Counter[str] = Counter({"selected": 0, "rejected": 0})
    by_reason_code: Counter[str] = Counter()

    for record in audit_records:
        source_path = record.source.get("source_path", record.source.get("path"))
        by_input_file[str(source_path) if source_path is not None else "<unknown>"] += 1
        by_source_format[record.source_format or "invalid"] += 1
        by_modality[record.modality or "invalid"] += 1
        if record.turn_count is None:
            turn_shape = "invalid"
        elif record.turn_count <= 1:
            turn_shape = "single_turn"
        else:
            turn_shape = "multi_turn"
        by_turn_shape[turn_shape] += 1
        by_disposition[record.disposition] += 1
        if record.reason_code is not None:
            by_reason_code[record.reason_code] += 1

    total = len(audit_records)
    selected = by_disposition["selected"]
    rejected = by_disposition["rejected"]
    return {
        "total": total,
        "selected": selected,
        "rejected": rejected,
        "totals": {
            "records": total,
            "selected": selected,
            "rejected": rejected,
        },
        "by_input_file": _sorted_counts(by_input_file),
        "by_source_format": _sorted_counts(by_source_format),
        "by_modality": _sorted_counts(by_modality),
        "by_turn_shape": _sorted_counts(by_turn_shape),
        "by_disposition": _sorted_counts(by_disposition),
        "by_reason_code": _sorted_counts(by_reason_code),
    }


def stage_dpo_artifacts(
    training_rows: Sequence[Mapping[str, Any]],
    audit_records: Sequence[AuditRecord],
    summary: Mapping[str, Any],
    warnings: Sequence[str],
    manifest_data: Mapping[str, Any],
    *,
    output_name: str,
    staging_dir: Path,
) -> dict[str, Path]:
    """Write a complete, self-describing artifact set beneath a staging dir."""
    _validate_output_name(output_name)
    stage = Path(staging_dir).resolve()
    try:
        stage.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DpoReportError(f"cannot create staging directory {stage}") from exc

    names = _artifact_names(output_name)
    staged = {name: _safe_child(stage, name) for name in names}

    strict_rows = [_validate_training_row(row) for row in training_rows]
    audit_values = [_audit_to_json(record) for record in audit_records]
    rejected_values = [
        value
        for record, value in zip(audit_records, audit_values, strict=True)
        if record.disposition == "rejected"
    ]
    safe_summary = _json_compatible(summary)
    if not isinstance(safe_summary, dict):
        raise DpoReportError("summary must be a JSON object")
    _validate_summary(safe_summary, audit_records)
    safe_warnings = [_redact_text(str(item)) for item in warnings]

    _write_jsonl(staged[output_name], strict_rows)
    _write_jsonl(staged["audit_records.jsonl"], audit_values)
    _write_jsonl(staged["rejected_records.jsonl"], rejected_values)
    _write_json(staged["summary.json"], safe_summary)
    _write_text(
        staged["warnings.log"],
        "".join(f"{warning}\n" for warning in safe_warnings),
    )

    non_manifest_stats = {
        name: compute_artifact_stat(
            staged[name], jsonl=name == output_name or name in _JSONL_FIXED_NAMES
        )
        for name in names
        if name != "manifest.json"
    }
    safe_manifest_data = redact_config(manifest_data)
    if not isinstance(safe_manifest_data, dict):
        raise DpoReportError("manifest_data must be a JSON object")
    manifest = dict(safe_manifest_data)
    counts = manifest.get("counts", {})
    if not isinstance(counts, dict):
        raise DpoReportError("manifest counts must be a JSON object")
    counts = dict(counts)
    counts.update(
        {
            "audit": len(audit_records),
            "selected": len(strict_rows),
            "rejected": len(rejected_values),
        }
    )
    manifest.update(
        {
            "schema_version": DPO_MANIFEST_SCHEMA_VERSION,
            "output_name": output_name,
            "counts": counts,
            "artifacts": {
                name: _stat_to_json(non_manifest_stats[name])
                for name in names
                if name != "manifest.json"
            },
        }
    )
    _write_json(staged["manifest.json"], manifest)
    return staged


def validate_staged_artifacts(
    staged: Mapping[str, Path], *, output_name: str
) -> dict[str, ArtifactStat]:
    """Strictly validate staged content and reconcile it with its manifest."""
    _validate_output_name(output_name)
    names = _artifact_names(output_name)
    if set(staged) != set(names):
        raise DpoReportError(
            f"staged artifact names must be exactly {', '.join(names)}"
        )
    paths = {name: Path(staged[name]) for name in names}
    for name, path in paths.items():
        if path.name != name or not path.is_file():
            raise DpoReportError(f"missing or unsafe staged artifact: {name}")

    stats = {
        name: compute_artifact_stat(
            paths[name], jsonl=name == output_name or name in _JSONL_FIXED_NAMES
        )
        for name in names
    }
    manifest = _read_json(paths["manifest.json"])
    _validate_manifest(
        manifest,
        output_name=output_name,
        actual_stats=stats,
    )
    _validate_artifact_contents(paths, output_name=output_name)
    _assert_no_sensitive_values(manifest, context="manifest.json")
    _assert_no_unredacted_warning(paths["warnings.log"])
    return stats


def recover_incomplete_publication(output_dir: Path, *, output_name: str) -> None:
    """Recover every durable publication journal under the output directory."""
    _validate_output_name(output_name)
    output = Path(output_dir).resolve()
    journal_root = _safe_child(output, ".dpo_publish")
    if not journal_root.exists():
        return
    if not journal_root.is_dir() or journal_root.is_symlink():
        raise DpoReportError(f"invalid DPO publication journal root: {journal_root}")

    for transaction in sorted(journal_root.iterdir(), key=lambda item: item.name):
        if not transaction.is_dir() or transaction.is_symlink():
            raise DpoReportError(f"invalid DPO publication journal entry: {transaction}")
        _validate_run_id(transaction.name)
        journal_path = transaction / "journal.json"
        if not journal_path.exists():
            # Delivery replacements never begin before journal.json exists.  A
            # journal-less directory is therefore either a pre-publication
            # crash or an interrupted cleanup of an already verified set.
            verify_committed_artifacts(output, output_name=output_name)
            _remove_transaction(transaction, journal_root)
            continue
        journal = _read_json(journal_path)
        _validate_journal(journal, transaction=transaction, output_name=output_name)

        if journal["state"] == "committed":
            verify_committed_artifacts(output, output_name=output_name)
            _remove_transaction(transaction, journal_root)
            continue

        if journal["state"] == "rolled_back":
            previous_manifest = _read_json(transaction / "old_manifest.json")
            current = verify_committed_artifacts(output, output_name=output_name)
            if current != previous_manifest:
                _restore_previous_set(output, transaction, journal)
                _verify_restored_set(output, output_name=output_name, journal=journal)
            _remove_transaction(transaction, journal_root)
            continue

        expected_manifest = _read_json(transaction / "expected_manifest.json")
        current_is_expected = False
        try:
            current = verify_committed_artifacts(output, output_name=output_name)
            current_is_expected = current == expected_manifest
        except DpoReportError:
            current_is_expected = False
        if current_is_expected:
            journal["state"] = "committed"
            _write_journal(transaction, journal)
            _remove_transaction(transaction, journal_root)
            continue

        _restore_previous_set(output, transaction, journal)
        _verify_restored_set(output, output_name=output_name, journal=journal)
        journal["state"] = "rolled_back"
        _write_journal(transaction, journal)
        _remove_transaction(transaction, journal_root)


def publish_staged_artifacts(
    staged: Mapping[str, Path],
    stats: Mapping[str, ArtifactStat],
    *,
    output_dir: Path,
    output_name: str,
    run_id: str,
) -> dict[str, Path]:
    """Publish a verified set with output-volume copies and manifest last."""
    _validate_output_name(output_name)
    _validate_run_id(run_id)
    names = _artifact_names(output_name)
    if set(staged) != set(names) or set(stats) != set(names):
        raise DpoReportError("staged paths and stats must describe the complete artifact set")
    training_stat = stats[output_name]
    if training_stat.line_count is None or training_stat.line_count == 0:
        raise DpoReportError("cannot publish a DPO dataset with zero selected samples")

    output = Path(output_dir).resolve()
    if output.exists() and not output.is_dir():
        raise DpoReportError(f"DPO output path is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise DpoReportError(f"DPO output directory may not be a symlink: {output}")

    recover_incomplete_publication(output, output_name=output_name)
    verify_committed_artifacts(output, output_name=output_name)

    # Revalidate the caller's staging claim before creating a transaction.
    current_stats = validate_staged_artifacts(staged, output_name=output_name)
    if dict(stats) != current_stats:
        raise DpoReportError("staged artifacts changed after validation")

    journal_root = _safe_child(output, ".dpo_publish")
    journal_root.mkdir(parents=True, exist_ok=True)
    if journal_root.is_symlink() or journal_root.resolve() != journal_root:
        raise DpoReportError("publication journal escaped the resolved output directory")
    transaction = _safe_child(journal_root, run_id)
    if transaction.exists():
        raise DpoReportError(f"publication run already exists: {run_id}")
    new_dir = transaction / "new"
    backup_dir = transaction / "backups"
    new_dir.mkdir(parents=True)
    backup_dir.mkdir()

    destinations = {name: _safe_child(output, name) for name in names}
    journal: dict[str, Any] | None = None
    try:
        for name in names:
            source = Path(staged[name])
            copied = _safe_child(new_dir, name)
            _copy_file_fsync(source, copied)
            copied_stat = compute_artifact_stat(
                copied, jsonl=name == output_name or name in _JSONL_FIXED_NAMES
            )
            if copied_stat != stats[name]:
                raise DpoReportError(
                    f"output-volume copy verification failed for {name}"
                )

        expected_manifest = _read_json(new_dir / "manifest.json")
        _write_json(transaction / "expected_manifest.json", expected_manifest)

        prior_files: dict[str, dict[str, Any]] = {}
        for name in names:
            destination = destinations[name]
            if destination.exists() and not destination.is_file():
                raise DpoReportError(f"managed output is not a file: {destination}")
            existed = destination.is_file()
            prior_files[name] = {"existed": existed, "backup": name if existed else None}
            if existed:
                _copy_file_fsync(destination, _safe_child(backup_dir, name))

        old_manifest = (
            _read_json(destinations["manifest.json"])
            if destinations["manifest.json"].is_file()
            else None
        )
        _write_json(transaction / "old_manifest.json", old_manifest)
        journal = {
            "schema_version": DPO_MANIFEST_SCHEMA_VERSION,
            "run_id": run_id,
            "output_name": output_name,
            "state": "prepared",
            "files": prior_files,
            "replaced": [],
        }
        _write_journal(transaction, journal)
        _fsync_dir(transaction)

        for name in names:  # manifest.json is deliberately last in this tuple.
            os.replace(_safe_child(new_dir, name), destinations[name])
            _fsync_file(destinations[name])
            _fsync_dir(output)
            journal["replaced"].append(name)
            _write_journal(transaction, journal)

        committed = verify_committed_artifacts(output, output_name=output_name)
        if committed != expected_manifest:
            raise DpoReportError("published manifest differs from the staged manifest")
        journal["state"] = "committed"
        _write_journal(transaction, journal)
        _remove_transaction(transaction, journal_root)
        return destinations
    except BaseException:
        if journal is not None and transaction.exists():
            try:
                _restore_previous_set(output, transaction, journal)
                _verify_restored_set(output, output_name=output_name, journal=journal)
                _remove_transaction(transaction, journal_root)
            except BaseException:
                # Keep the durable journal so startup recovery can retry safely.
                pass
        else:
            _remove_unprepared_transaction(transaction, journal_root)
        raise


def verify_committed_artifacts(
    output_dir: Path, *, output_name: str
) -> dict[str, Any] | None:
    """Verify the complete fixed-name committed set, or return None if absent."""
    _validate_output_name(output_name)
    output = Path(output_dir).resolve()
    names = _artifact_names(output_name)
    if not output.exists():
        return None
    if not output.is_dir():
        raise DpoReportError(f"DPO output path is not a directory: {output}")
    paths = {name: _safe_child(output, name) for name in names}
    manifest_path = paths["manifest.json"]
    if not manifest_path.is_file():
        orphans = [name for name in names if paths[name].exists()]
        if orphans:
            raise DpoReportError(
                "managed DPO artifacts exist without manifest.json: "
                + ", ".join(orphans)
            )
        return None

    manifest = _read_json(manifest_path)
    actual_stats: dict[str, ArtifactStat] = {}
    for name in names:
        path = paths[name]
        if not path.is_file():
            raise DpoReportError(f"committed artifact is missing: {name}")
        if name != "manifest.json":
            actual_stats[name] = compute_artifact_stat(
                path, jsonl=name == output_name or name in _JSONL_FIXED_NAMES
            )
    _validate_manifest(
        manifest,
        output_name=output_name,
        actual_stats=actual_stats,
    )
    _validate_artifact_contents(paths, output_name=output_name)
    _assert_no_sensitive_values(manifest, context="manifest.json")
    _assert_no_unredacted_warning(paths["warnings.log"])
    return manifest


def write_failed_run(
    work_dir: Path,
    run_id: str,
    audit_records: Sequence[AuditRecord],
    summary: Mapping[str, Any],
    warnings: Sequence[str],
    error: str,
) -> Path:
    """Persist zero-result/failure diagnostics away from delivery artifacts."""
    _validate_run_id(run_id)
    work = Path(work_dir).resolve()
    failed_root = _safe_child(work, "failed_runs")
    failed = _safe_child(failed_root, run_id)
    if failed.exists():
        raise DpoReportError(f"failed-run diagnostics already exist: {failed}")
    failed.mkdir(parents=True)
    audit_values = [_audit_to_json(record) for record in audit_records]
    rejected_values = [
        value
        for record, value in zip(audit_records, audit_values, strict=True)
        if record.disposition == "rejected"
    ]
    _write_jsonl(failed / "audit_records.jsonl", audit_values)
    _write_jsonl(failed / "rejected_records.jsonl", rejected_values)
    _write_json(failed / "summary.json", _json_compatible(summary))
    _write_text(
        failed / "warnings.log",
        "".join(f"{_redact_text(str(item))}\n" for item in warnings),
    )
    _write_json(
        failed / "error.json",
        {"run_id": run_id, "error": _redact_text(str(error))},
    )
    return failed


def _artifact_names(output_name: str) -> tuple[str, ...]:
    return (output_name, *_FIXED_ARTIFACT_NAMES)


def _validate_output_name(output_name: str) -> None:
    if not isinstance(output_name, str) or not output_name:
        raise DpoReportError("output_name must be a nonempty string")
    if (
        Path(output_name).name != output_name
        or "/" in output_name
        or "\\" in output_name
        or output_name in {".", ".."}
        or output_name.endswith((" ", "."))
        or not output_name.lower().endswith(".jsonl")
        or output_name.casefold() in {name.casefold() for name in _RESERVED_OUTPUT_NAMES}
    ):
        raise DpoReportError(f"unsafe DPO output_name: {output_name!r}")
    stem = output_name.split(".", 1)[0].rstrip(" .").casefold()
    if stem in _WINDOWS_DEVICE_NAMES or ":" in output_name:
        raise DpoReportError(f"unsafe DPO output_name: {output_name!r}")


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _RUN_ID_RE.fullmatch(run_id):
        raise DpoReportError(f"unsafe publication run_id: {run_id!r}")
    if run_id in {".", ".."} or run_id.endswith((" ", ".")):
        raise DpoReportError(f"unsafe publication run_id: {run_id!r}")


def _safe_child(parent: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise DpoReportError(f"unsafe artifact path component: {name!r}")
    resolved_parent = parent.resolve()
    candidate = resolved_parent / name
    if candidate.parent.resolve() != resolved_parent:
        raise DpoReportError(f"artifact path escapes its parent: {name!r}")
    return candidate


def _redact_text(text: str) -> str:
    if text.lstrip().lower().startswith("data:"):
        return "<redacted-data-url>"
    text = _DATA_URL_RE.sub("<redacted-data-url>", text)
    text = _BEARER_RE.sub("<redacted-authorization>", text)
    text = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}{match.group(2)}<redacted>", text
    )
    return text


def _is_secret_key(key: str) -> bool:
    collapsed = re.sub(r"[^a-z0-9]", "", key.casefold())
    return any(
        collapsed.endswith(suffix)
        for suffix in (
            "apikey",
            "authorization",
            "accesstoken",
            "authtoken",
            "bearertoken",
            "clientsecret",
            "password",
            "secret",
        )
    )


def _json_compatible(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_compatible(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _audit_to_json(record: AuditRecord) -> dict[str, Any]:
    if not isinstance(record, AuditRecord):
        raise DpoReportError("audit_records must contain AuditRecord values")
    if record.disposition == "selected" and record.reason_code is not None:
        raise DpoReportError("a selected audit record cannot have a rejection reason")
    if record.disposition == "rejected" and record.reason_code is None:
        raise DpoReportError("a rejected audit record requires a stable reason_code")
    value = _json_compatible(asdict(record))
    assert isinstance(value, dict)
    return value


def _audit_from_json(value: Any) -> AuditRecord:
    fields = {
        "source",
        "sample_id",
        "source_format",
        "turn_index",
        "turn_count",
        "modality",
        "digests",
        "prediction",
        "inference_error",
        "judge",
        "disposition",
        "reason_code",
        "detail",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise DpoReportError("audit record has an invalid field set")
    if not isinstance(value["source"], dict) or not isinstance(value["digests"], dict):
        raise DpoReportError("audit source and digests must be JSON objects")
    if not all(isinstance(item, str) for item in value["digests"].values()):
        raise DpoReportError("audit digest values must be strings")
    if value["sample_id"] is not None and not isinstance(value["sample_id"], str):
        raise DpoReportError("audit sample_id must be a string or null")
    if value["source_format"] not in {None, "alpaca", "sharegpt"}:
        raise DpoReportError("audit source_format is invalid")
    if value["modality"] not in {None, "text", "single_image", "multi_image"}:
        raise DpoReportError("audit modality is invalid")
    for field in ("turn_index", "turn_count"):
        item = value[field]
        if item is not None and (type(item) is not int or item < 0):
            raise DpoReportError(f"audit {field} must be a nonnegative integer or null")
    for field in ("prediction", "inference_error"):
        if value[field] is not None and not isinstance(value[field], str):
            raise DpoReportError(f"audit {field} must be a string or null")
    if value["judge"] is not None and not isinstance(value["judge"], dict):
        raise DpoReportError("audit judge must be an object or null")
    if value["disposition"] not in {"selected", "rejected"}:
        raise DpoReportError("audit disposition is invalid")
    if value["reason_code"] is not None and value["reason_code"] not in get_args(ReasonCode):
        raise DpoReportError("audit reason_code is invalid")
    if not isinstance(value["detail"], str):
        raise DpoReportError("audit detail must be a string")
    record = AuditRecord(**value)
    _audit_to_json(record)
    return record


def _validate_summary(
    summary: Mapping[str, Any], audit_records: Sequence[AuditRecord]
) -> None:
    expected = build_summary(audit_records)
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            raise DpoReportError(f"summary aggregate does not match audit records: {key}")


def _validate_artifact_contents(
    paths: Mapping[str, Path], *, output_name: str
) -> None:
    training_rows = _read_jsonl(paths[output_name])
    for row in training_rows:
        _validate_training_row(row)
    audit_values = _read_jsonl(paths["audit_records.jsonl"])
    audits = [_audit_from_json(value) for value in audit_values]
    rejected_values = _read_jsonl(paths["rejected_records.jsonl"])
    expected_rejected = [
        value
        for value, record in zip(audit_values, audits, strict=True)
        if record.disposition == "rejected"
    ]
    if rejected_values != expected_rejected:
        raise DpoReportError("rejected_records.jsonl does not match rejected audit records")
    selected_count = sum(record.disposition == "selected" for record in audits)
    if len(training_rows) != selected_count:
        raise DpoReportError("training row count does not match selected audit records")
    summary = _read_json(paths["summary.json"])
    if not isinstance(summary, dict):
        raise DpoReportError("summary.json must contain a JSON object")
    _validate_summary(summary, audits)


def _validate_training_row(row: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise DpoReportError("training rows must be JSON objects")
    required = {"conversations", "chosen", "rejected", "images"}
    if set(row) != required:
        raise DpoReportError(
            "training rows must contain exactly conversations, chosen, rejected, images"
        )
    value = _json_compatible(row)
    if not isinstance(value["conversations"], list):
        raise DpoReportError("training conversations must be a JSON array")
    for turn in value["conversations"]:
        if (
            not isinstance(turn, dict)
            or set(turn) != {"from", "value"}
            or turn["from"] not in {"human", "gpt"}
            or not isinstance(turn["value"], str)
        ):
            raise DpoReportError("training conversations contain an invalid turn")
    if not isinstance(value["chosen"], str) or not isinstance(value["rejected"], str):
        raise DpoReportError("training chosen and rejected values must be strings")
    if not isinstance(value["images"], list) or not all(
        isinstance(item, str) for item in value["images"]
    ):
        raise DpoReportError("training images must be an array of strings")
    return value


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _json_dumps(value: Any, *, indent: int | None = None) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            sort_keys=indent is not None,
            separators=(",", ":") if indent is None else None,
        )
    except (TypeError, ValueError) as exc:
        raise DpoReportError(f"value is not strict JSON: {exc}") from exc


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    _write_text(path, "".join(f"{_json_dumps(value)}\n" for value in values))


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, _json_dumps(value, indent=2) + "\n")


def _write_text(path: Path, text: str) -> None:
    data = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def _strict_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_strict_object,
        parse_constant=_reject_constant,
    )


def _read_json(path: Path) -> Any:
    try:
        return _strict_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DpoReportError(f"invalid JSON artifact {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[Any]:
    try:
        data = path.read_bytes()
        _validate_jsonl_bytes(data)
        return [_strict_loads(line.decode("utf-8")) for line in data.splitlines()]
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise DpoReportError(f"invalid JSONL artifact {path}: {exc}") from exc


def _validate_jsonl_bytes(data: bytes) -> int:
    if not data:
        return 0
    if not data.endswith(b"\n"):
        raise ValueError("JSONL must end with a newline")
    lines = data.splitlines()
    for index, line in enumerate(lines, start=1):
        if not line:
            raise ValueError(f"blank JSONL line {index}")
        value = _strict_loads(line.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {index} is not an object")
    return len(lines)


def _stat_to_json(stat: ArtifactStat) -> dict[str, Any]:
    return {
        "sha256": stat.sha256,
        "byte_count": stat.byte_count,
        "line_count": stat.line_count,
    }


def _parse_stat(value: Any, *, name: str) -> ArtifactStat:
    if not isinstance(value, dict) or set(value) != {
        "sha256",
        "byte_count",
        "line_count",
    }:
        raise DpoReportError(f"invalid manifest artifact stat for {name}")
    sha256 = value["sha256"]
    byte_count = value["byte_count"]
    line_count = value["line_count"]
    if (
        not isinstance(sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", sha256)
        or type(byte_count) is not int
        or byte_count < 0
        or (line_count is not None and (type(line_count) is not int or line_count < 0))
    ):
        raise DpoReportError(f"invalid manifest artifact stat for {name}")
    return ArtifactStat(sha256, byte_count, line_count)


def _validate_manifest(
    manifest: Any,
    *,
    output_name: str,
    actual_stats: Mapping[str, ArtifactStat],
) -> None:
    if not isinstance(manifest, dict):
        raise DpoReportError("manifest.json must contain a JSON object")
    if manifest.get("schema_version") != DPO_MANIFEST_SCHEMA_VERSION:
        raise DpoReportError("unsupported DPO manifest schema version")
    if manifest.get("output_name") != output_name:
        raise DpoReportError("manifest output_name does not match the requested output")
    artifacts = manifest.get("artifacts")
    expected_names = set(_artifact_names(output_name)) - {"manifest.json"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise DpoReportError("manifest must describe every non-manifest artifact exactly once")
    for name in expected_names:
        expected = _parse_stat(artifacts[name], name=name)
        if name not in actual_stats or expected != actual_stats[name]:
            raise DpoReportError(f"committed artifact does not match manifest: {name}")
    if "manifest.json" in artifacts:
        raise DpoReportError("manifest.json may not hash itself")


def _assert_no_sensitive_values(value: Any, *, context: str, key: str = "") -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            normalized = str(child_key).replace("-", "_")
            if _is_secret_key(normalized) and child != "<redacted>":
                raise DpoReportError(f"unredacted secret in {context}")
            _assert_no_sensitive_values(child, context=context, key=str(child_key))
    elif isinstance(value, list):
        for child in value:
            _assert_no_sensitive_values(child, context=context, key=key)
    elif isinstance(value, str):
        if _DATA_URL_RE.search(value) or _BEARER_RE.search(value):
            raise DpoReportError(f"unredacted sensitive value in {context}")


def _assert_no_unredacted_warning(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise DpoReportError(f"invalid warnings artifact {path}") from exc
    if _DATA_URL_RE.search(text) or _BEARER_RE.search(text):
        raise DpoReportError("warnings.log contains an unredacted sensitive value")
    for match in _SECRET_ASSIGNMENT_RE.finditer(text):
        if "<redacted>" not in match.group(0):
            raise DpoReportError("warnings.log contains an unredacted secret")


def _copy_file_fsync(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise DpoReportError(f"artifact source is not a file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _fsync_file(path: Path) -> None:
    # Windows rejects FlushFileBuffers on a descriptor opened read-only.
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _write_journal(transaction: Path, journal: Mapping[str, Any]) -> None:
    run_id = transaction.name
    _validate_run_id(run_id)
    path = transaction / "journal.json"
    temporary = transaction / f"journal.{run_id}.tmp"
    if temporary.exists():
        temporary.unlink()
    _write_json(temporary, journal)
    os.replace(temporary, path)
    _fsync_file(path)
    _fsync_dir(transaction)


def _validate_journal(
    journal: Any, *, transaction: Path, output_name: str
) -> None:
    if not isinstance(journal, dict):
        raise DpoReportError(f"invalid publication journal: {transaction}")
    if (
        journal.get("schema_version") != DPO_MANIFEST_SCHEMA_VERSION
        or journal.get("run_id") != transaction.name
        or journal.get("output_name") != output_name
        or journal.get("state") not in {"prepared", "committed", "rolled_back"}
        or not isinstance(journal.get("files"), dict)
        or set(journal["files"]) != set(_artifact_names(output_name))
        or not isinstance(journal.get("replaced"), list)
    ):
        raise DpoReportError(f"invalid publication journal: {transaction}")
    for name, entry in journal["files"].items():
        if (
            not isinstance(entry, dict)
            or set(entry) != {"existed", "backup"}
            or type(entry["existed"]) is not bool
            or entry["backup"] != (name if entry["existed"] else None)
        ):
            raise DpoReportError(f"invalid publication journal file entry: {name}")
    if any(name not in journal["files"] for name in journal["replaced"]):
        raise DpoReportError(f"invalid publication replacement progress: {transaction}")


def _restore_previous_set(
    output: Path, transaction: Path, journal: Mapping[str, Any]
) -> None:
    names = _artifact_names(journal["output_name"])
    manifest_destination = _safe_child(output, "manifest.json")
    if manifest_destination.exists():
        if not manifest_destination.is_file():
            raise DpoReportError("cannot remove non-file manifest during recovery")
        manifest_destination.unlink()
        _fsync_dir(output)

    for name in names:
        if name == "manifest.json":
            continue
        _restore_one(output, transaction, journal, name)
    _restore_one(output, transaction, journal, "manifest.json")
    _fsync_dir(output)


def _restore_one(
    output: Path, transaction: Path, journal: Mapping[str, Any], name: str
) -> None:
    destination = _safe_child(output, name)
    entry = journal["files"][name]
    if entry["existed"]:
        backup = _safe_child(transaction / "backups", name)
        if not backup.is_file():
            raise DpoReportError(f"publication backup is missing: {name}")
        temporary = _safe_child(output, f".{journal['run_id']}.restore.{name}")
        if temporary.exists():
            if not temporary.is_file():
                raise DpoReportError(f"unsafe recovery temporary: {temporary}")
            temporary.unlink()
        _copy_file_fsync(backup, temporary)
        os.replace(temporary, destination)
        _fsync_file(destination)
    elif destination.exists():
        if not destination.is_file():
            raise DpoReportError(f"cannot remove non-file managed artifact: {destination}")
        destination.unlink()


def _verify_restored_set(
    output: Path, *, output_name: str, journal: Mapping[str, Any]
) -> None:
    previous_manifest = _read_json(Path(output) / ".dpo_publish" / journal["run_id"] / "old_manifest.json")
    current = verify_committed_artifacts(output, output_name=output_name)
    if current != previous_manifest:
        raise DpoReportError("restored DPO artifact set does not match its old manifest")


def _remove_transaction(transaction: Path, journal_root: Path) -> None:
    if transaction.exists():
        shutil.rmtree(transaction)
    try:
        journal_root.rmdir()
    except OSError:
        pass


def _remove_unprepared_transaction(transaction: Path, journal_root: Path) -> None:
    try:
        _remove_transaction(transaction, journal_root)
    except OSError:
        pass
