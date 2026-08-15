from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from eval_tool import dpo_report
from eval_tool.dpo_report import (
    ArtifactStat,
    AuditRecord,
    DpoReportError,
    build_summary,
    compute_artifact_stat,
    publish_staged_artifacts,
    recover_incomplete_publication,
    stage_dpo_artifacts,
    validate_staged_artifacts,
    verify_committed_artifacts,
    write_failed_run,
)


def _source(
    path: str = "input.jsonl", *, record_index: int = 0, line_number: int = 1
) -> dict[str, object]:
    return {
        "source_path": path,
        "input_index": 0,
        "container_format": "jsonl",
        "record_index": record_index,
        "line_number": line_number,
        "source_id": f"source-{record_index}",
        "raw_digest": f"raw-{record_index}",
    }


def _audit(
    *,
    sample_id: str | None = "sample-0",
    source_format: str | None = "alpaca",
    turn_index: int | None = 0,
    turn_count: int | None = 1,
    modality: str | None = "text",
    disposition: str = "selected",
    reason_code: str | None = None,
    source: dict[str, object] | None = None,
    prediction: str | None = "wrong answer",
    inference_error: str | None = None,
    judge: dict[str, object] | None = None,
) -> AuditRecord:
    return AuditRecord(
        source=source or _source(),
        sample_id=sample_id,
        source_format=source_format,  # type: ignore[arg-type]
        turn_index=turn_index,
        turn_count=turn_count,
        modality=modality,  # type: ignore[arg-type]
        digests={
            "conversation": "conversation-digest",
            "chosen": "chosen-digest",
            "images": "images-digest",
        }
        if sample_id is not None
        else {},
        prediction=prediction,
        inference_error=inference_error,
        judge=judge,
        disposition=disposition,  # type: ignore[arg-type]
        reason_code=reason_code,  # type: ignore[arg-type]
        detail="kept" if disposition == "selected" else "not selected",
    )


def _row(*, rejected: str = "wrong answer") -> dict[str, object]:
    return {
        "conversations": [{"from": "human", "value": "question"}],
        "chosen": {"from": "gpt", "value": "reference answer"},
        "rejected": {"from": "gpt", "value": rejected},
        "images": ["relative/image.png"],
    }


def _manifest_data() -> dict[str, object]:
    return {
        "effective_config": {
            "inputs": ["input.jsonl"],
            "infer": {"enable_thinking": True, "max_new_tokens": 64},
        },
        "inputs": [
            {
                "path": "input.jsonl",
                "sha256": "a" * 64,
                "byte_count": 123,
            }
        ],
        "normalizer_version": "dpo-normalizer-v1",
        "model": {
            "name": "fake-local-model",
            "resolved_path": "C:/models/fake",
            "checkpoint_identity": "checkpoint-1",
            "enable_thinking": True,
            "generation": {"max_new_tokens": 64, "batch_size": 1},
        },
        "fingerprints": {
            "inference": "infer-fp",
            "judge_request": "judge-request-fp",
            "judge_parse": "judge-parse-fp",
        },
        "counts": {"input": 1, "selected": 1, "rejected": 0},
    }


def _stage(
    root: Path,
    *,
    output_name: str = "training.jsonl",
    rejected: str = "wrong answer",
    warnings: list[str] | None = None,
) -> tuple[dict[str, Path], dict[str, ArtifactStat]]:
    audit = [_audit(prediction=rejected)]
    staged = stage_dpo_artifacts(
        [_row(rejected=rejected)],
        audit,
        build_summary(audit),
        warnings or [],
        _manifest_data(),
        output_name=output_name,
        staging_dir=root,
    )
    return staged, validate_staged_artifacts(staged, output_name=output_name)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_training_rows_have_exactly_conversations_chosen_rejected_images(
    tmp_path: Path,
) -> None:
    staged, _ = _stage(tmp_path / "stage")
    assert set(_read_jsonl(staged["training.jsonl"])[0]) == {
        "conversations",
        "chosen",
        "rejected",
        "images",
    }

    with pytest.raises(DpoReportError, match="exactly"):
        stage_dpo_artifacts(
            [{**_row(), "sample_id": "must-not-leak"}],
            [_audit()],
            {},
            [],
            {},
            output_name="training.jsonl",
            staging_dir=tmp_path / "invalid",
        )


@pytest.mark.parametrize("field", ["chosen", "rejected"])
def test_training_rows_reject_legacy_string_preference_answers(field: str) -> None:
    row = _row()
    row[field] = "legacy string answer"

    with pytest.raises(DpoReportError, match="ShareGPT assistant message"):
        dpo_report._validate_training_row(row)


def test_invalid_valid_json_record_appears_in_audit_and_rejected(
    tmp_path: Path,
) -> None:
    invalid = _audit(
        sample_id=None,
        source_format=None,
        turn_index=None,
        turn_count=None,
        modality=None,
        disposition="rejected",
        reason_code="invalid_schema",
        prediction=None,
    )
    staged = stage_dpo_artifacts(
        [_row()],
        [_audit(), invalid],
        build_summary([_audit(), invalid]),
        [],
        _manifest_data(),
        output_name="training.jsonl",
        staging_dir=tmp_path / "stage",
    )
    assert _read_jsonl(staged["audit_records.jsonl"])[1]["sample_id"] is None
    assert _read_jsonl(staged["rejected_records.jsonl"])[0]["reason_code"] == "invalid_schema"


def test_every_duplicate_and_conflict_source_has_its_own_disposition(
    tmp_path: Path,
) -> None:
    records = [
        _audit(sample_id="selected-0"),
        _audit(
            sample_id="duplicate-1",
            source=_source(record_index=1, line_number=2),
            disposition="rejected",
            reason_code="duplicate",
        ),
        _audit(
            sample_id="conflict-1",
            source=_source(record_index=2, line_number=3),
            disposition="rejected",
            reason_code="reference_conflict",
        ),
        _audit(
            sample_id="conflict-2",
            source=_source(record_index=3, line_number=4),
            disposition="rejected",
            reason_code="reference_conflict",
        ),
    ]
    staged = stage_dpo_artifacts(
        [_row()],
        records,
        build_summary(records),
        [],
        _manifest_data(),
        output_name="training.jsonl",
        staging_dir=tmp_path / "stage",
    )
    rejected = _read_jsonl(staged["rejected_records.jsonl"])
    assert [item["sample_id"] for item in rejected] == [
        "duplicate-1",
        "conflict-1",
        "conflict-2",
    ]


def test_summary_counts_match_audit_and_rejected_by_stable_reason() -> None:
    records = [
        _audit(),
        _audit(disposition="rejected", reason_code="duplicate"),
        _audit(disposition="rejected", reason_code="judge_pass"),
    ]
    summary = build_summary(records)
    assert summary["total"] == 3
    assert summary["by_disposition"] == {"rejected": 2, "selected": 1}
    assert summary["by_reason_code"] == {"duplicate": 1, "judge_pass": 1}
    assert sum(summary["by_disposition"].values()) == summary["total"]


def test_summary_groups_by_input_format_modality_turn_shape_and_reason() -> None:
    records = [
        _audit(source_format="alpaca", modality="text", turn_count=1),
        _audit(
            source_format="sharegpt",
            modality="single_image",
            turn_count=3,
            disposition="rejected",
            reason_code="judge_pass",
            source=_source("other.json"),
        ),
        _audit(
            sample_id=None,
            source_format=None,
            modality=None,
            turn_count=None,
            turn_index=None,
            disposition="rejected",
            reason_code="invalid_schema",
            prediction=None,
        ),
    ]
    summary = build_summary(records)
    assert summary["by_input_file"] == {"input.jsonl": 2, "other.json": 1}
    assert summary["by_source_format"] == {"alpaca": 1, "invalid": 1, "sharegpt": 1}
    assert summary["by_modality"] == {
        "invalid": 1,
        "multi_image": 0,
        "single_image": 1,
        "text": 1,
    }
    assert summary["by_turn_shape"] == {
        "invalid": 1,
        "multi_turn": 1,
        "single_turn": 1,
    }


def test_manifest_records_sha256_byte_count_and_line_count(tmp_path: Path) -> None:
    staged, stats = _stage(tmp_path / "stage")
    manifest = json.loads(staged["manifest.json"].read_text(encoding="utf-8"))
    assert "manifest.json" not in manifest["artifacts"]
    for name, expected in manifest["artifacts"].items():
        actual = stats[name]
        assert expected == {
            "sha256": actual.sha256,
            "byte_count": actual.byte_count,
            "line_count": actual.line_count,
        }
    assert manifest["artifacts"]["training.jsonl"]["line_count"] == 1


def test_staging_collects_artifact_stats_while_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_post_write_scan(*args: object, **kwargs: object) -> ArtifactStat:
        raise AssertionError("staging must not rescan artifacts it just wrote")

    monkeypatch.setattr(dpo_report, "compute_artifact_stat", reject_post_write_scan)
    audit = [_audit()]

    staged = stage_dpo_artifacts(
        [_row()],
        audit,
        build_summary(audit),
        [],
        _manifest_data(),
        output_name="training.jsonl",
        staging_dir=tmp_path / "stage",
    )

    manifest = json.loads(staged["manifest.json"].read_text(encoding="utf-8"))
    assert manifest["artifacts"]["training.jsonl"]["line_count"] == 1


def test_manifest_and_warnings_recursively_redact_api_keys(tmp_path: Path) -> None:
    fake_secret = "FAKE_TEST_SECRET_DO_NOT_USE"
    manifest_data = _manifest_data()
    manifest_data["judge"] = {
        "headers": {"Authorization": f"Bearer {fake_secret}"},
        "nested": [{"api_key": fake_secret}],
        "image": f"data:image/png;base64,{fake_secret}",
    }
    staged = stage_dpo_artifacts(
        [_row()],
        [_audit()],
        build_summary([_audit()]),
        [
            f"api_key={fake_secret}",
            f"request Authorization: Bearer {fake_secret}",
            f"bad image data:image/png;base64,{fake_secret}",
        ],
        manifest_data,
        output_name="training.jsonl",
        staging_dir=tmp_path / "stage",
    )
    validate_staged_artifacts(staged, output_name="training.jsonl")
    manifest_text = staged["manifest.json"].read_text(encoding="utf-8")
    warnings_text = staged["warnings.log"].read_text(encoding="utf-8")
    assert fake_secret not in manifest_text
    assert fake_secret not in warnings_text
    assert "data:image" not in manifest_text
    assert "data:image" not in warnings_text
    assert "<redacted>" in manifest_text


def test_manifest_records_model_checkpoint_thinking_and_phase_fingerprints(
    tmp_path: Path,
) -> None:
    staged, _ = _stage(tmp_path / "stage")
    manifest = json.loads(staged["manifest.json"].read_text(encoding="utf-8"))
    assert manifest["inputs"][0]["sha256"] == "a" * 64
    assert manifest["model"]["checkpoint_identity"] == "checkpoint-1"
    assert manifest["model"]["enable_thinking"] is True
    assert manifest["fingerprints"] == {
        "inference": "infer-fp",
        "judge_parse": "judge-parse-fp",
        "judge_request": "judge-request-fp",
    }


def test_audit_fields_include_source_turn_digests_prediction_judge_and_disposition(
    tmp_path: Path,
) -> None:
    record = _audit(
        judge={
            "request_fingerprint": "request-fp",
            "parse_fingerprint": "parse-fp",
            "verdict": {"hit": 0},
        }
    )
    staged = stage_dpo_artifacts(
        [_row()],
        [record],
        build_summary([record]),
        [],
        _manifest_data(),
        output_name="training.jsonl",
        staging_dir=tmp_path / "stage",
    )
    value = _read_jsonl(staged["audit_records.jsonl"])[0]
    assert set(value) == {
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
    assert value["source"]["line_number"] == 1
    assert value["judge"]["verdict"] == {"hit": 0}


def test_configured_output_name_is_used_without_changing_other_artifact_names(
    tmp_path: Path,
) -> None:
    staged, _ = _stage(tmp_path / "stage", output_name="custom-dpo.jsonl")
    assert set(staged) == {
        "custom-dpo.jsonl",
        "audit_records.jsonl",
        "rejected_records.jsonl",
        "summary.json",
        "warnings.log",
        "manifest.json",
    }


def test_warnings_log_is_always_published(tmp_path: Path) -> None:
    staged, stats = _stage(tmp_path / "stage")
    published = publish_staged_artifacts(
        staged,
        stats,
        output_dir=tmp_path / "output",
        output_name="training.jsonl",
        run_id="run-warnings",
    )
    assert published["warnings.log"].is_file()
    assert published["warnings.log"].read_bytes() == b""


def test_publish_does_not_reparse_already_validated_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged, stats = _stage(tmp_path / "stage")

    def reject_duplicate_validation(*args: object, **kwargs: object):
        raise AssertionError("publish must verify the copy, not reparse staging")

    monkeypatch.setattr(
        dpo_report, "validate_staged_artifacts", reject_duplicate_validation
    )

    published = publish_staged_artifacts(
        staged,
        stats,
        output_dir=tmp_path / "output",
        output_name="training.jsonl",
        run_id="single-validation",
    )

    assert published["training.jsonl"].is_file()


def test_publish_full_verification_runs_before_but_not_after_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged, stats = _stage(tmp_path / "stage")
    real_verify = dpo_report.verify_committed_artifacts
    calls = 0

    def counting_verify(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(dpo_report, "verify_committed_artifacts", counting_verify)

    publish_staged_artifacts(
        staged,
        stats,
        output_dir=tmp_path / "output",
        output_name="training.jsonl",
        run_id="single-committed-verification",
    )

    assert calls == 1


def test_manifest_is_replaced_last(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    staged, stats = _stage(tmp_path / "stage")
    output = (tmp_path / "output").resolve()
    replaced: list[str] = []
    original_replace = dpo_report.os.replace

    def recording_replace(source: object, destination: object) -> None:
        destination_path = Path(destination)
        if destination_path.parent == output and destination_path.name in staged:
            replaced.append(destination_path.name)
        original_replace(source, destination)

    monkeypatch.setattr(dpo_report.os, "replace", recording_replace)
    publish_staged_artifacts(
        staged,
        stats,
        output_dir=output,
        output_name="training.jsonl",
        run_id="run-order",
    )
    assert replaced == [
        "training.jsonl",
        "audit_records.jsonl",
        "rejected_records.jsonl",
        "summary.json",
        "warnings.log",
        "manifest.json",
    ]


def test_publish_failure_restores_previous_committed_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    old_staged, old_stats = _stage(tmp_path / "old-stage", rejected="old prediction")
    publish_staged_artifacts(
        old_staged,
        old_stats,
        output_dir=output,
        output_name="training.jsonl",
        run_id="old-run",
    )
    old_bytes = {
        name: (output / name).read_bytes()
        for name in old_staged
    }
    new_staged, new_stats = _stage(tmp_path / "new-stage", rejected="new prediction")
    original_replace = dpo_report.os.replace
    failed = False

    def fail_once(source: object, destination: object) -> None:
        nonlocal failed
        destination_path = Path(destination)
        if destination_path == output.resolve() / "summary.json" and not failed:
            failed = True
            raise OSError("simulated replace failure")
        original_replace(source, destination)

    monkeypatch.setattr(dpo_report.os, "replace", fail_once)
    with pytest.raises(OSError, match="simulated replace failure"):
        publish_staged_artifacts(
            new_staged,
            new_stats,
            output_dir=output,
            output_name="training.jsonl",
            run_id="new-run",
        )
    verify_committed_artifacts(output, output_name="training.jsonl")
    assert {name: (output / name).read_bytes() for name in old_staged} == old_bytes


def test_incomplete_journal_recovers_previous_committed_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    old_staged, old_stats = _stage(tmp_path / "old-stage", rejected="old prediction")
    publish_staged_artifacts(
        old_staged,
        old_stats,
        output_dir=output,
        output_name="training.jsonl",
        run_id="base-run",
    )
    old_manifest = verify_committed_artifacts(output, output_name="training.jsonl")
    new_staged, new_stats = _stage(tmp_path / "new-stage", rejected="new prediction")
    original_replace = dpo_report.os.replace
    crashed = False

    def crash_after_first_delivery_replace(source: object, destination: object) -> None:
        nonlocal crashed
        destination_path = Path(destination)
        original_replace(source, destination)
        if destination_path == output.resolve() / "training.jsonl" and not crashed:
            crashed = True
            raise SystemExit("simulated hard interruption")

    def fail_in_process_restore(*args: object, **kwargs: object) -> None:
        raise OSError("recovery deferred to next startup")

    monkeypatch.setattr(dpo_report.os, "replace", crash_after_first_delivery_replace)
    monkeypatch.setattr(dpo_report, "_restore_previous_set", fail_in_process_restore)
    with pytest.raises(SystemExit, match="hard interruption"):
        publish_staged_artifacts(
            new_staged,
            new_stats,
            output_dir=output,
            output_name="training.jsonl",
            run_id="interrupted-run",
        )

    monkeypatch.setattr(dpo_report.os, "replace", original_replace)
    monkeypatch.undo()
    recover_incomplete_publication(output, output_name="training.jsonl")
    assert verify_committed_artifacts(output, output_name="training.jsonl") == old_manifest
    assert not (output / ".dpo_publish").exists()


def test_output_volume_temp_copy_is_rehashed_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged, stats = _stage(tmp_path / "stage")
    output = tmp_path / "output"
    original_copy = dpo_report._copy_file_fsync
    corrupted = False

    def corrupt_first_copy(source: Path, destination: Path) -> None:
        nonlocal corrupted
        original_copy(source, destination)
        if destination.name == "training.jsonl" and destination.parent.name == "new" and not corrupted:
            corrupted = True
            with destination.open("ab") as stream:
                stream.write(b"{}\n")
                stream.flush()
                os.fsync(stream.fileno())

    monkeypatch.setattr(dpo_report, "_copy_file_fsync", corrupt_first_copy)
    with pytest.raises(DpoReportError, match="copy verification"):
        publish_staged_artifacts(
            staged,
            stats,
            output_dir=output,
            output_name="training.jsonl",
            run_id="corrupt-copy-run",
        )
    assert verify_committed_artifacts(output, output_name="training.jsonl") is None


def test_committed_manifest_mismatch_is_rejected_on_next_start(tmp_path: Path) -> None:
    staged, stats = _stage(tmp_path / "stage")
    output = tmp_path / "output"
    publish_staged_artifacts(
        staged,
        stats,
        output_dir=output,
        output_name="training.jsonl",
        run_id="valid-run",
    )
    with (output / "training.jsonl").open("ab") as stream:
        stream.write(json.dumps(_row()).encode("utf-8") + b"\n")
    with pytest.raises(DpoReportError, match="does not match manifest"):
        verify_committed_artifacts(output, output_name="training.jsonl")


def test_existing_training_output_without_manifest_is_not_trusted_or_overwritten(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    orphan = output / "training.jsonl"
    orphan.write_text("legacy output\n", encoding="utf-8")
    staged, stats = _stage(tmp_path / "stage")
    with pytest.raises(DpoReportError, match="without manifest"):
        publish_staged_artifacts(
            staged,
            stats,
            output_dir=output,
            output_name="training.jsonl",
            run_id="refuse-orphan",
        )
    assert orphan.read_text(encoding="utf-8") == "legacy output\n"


def test_zero_selected_does_not_touch_output_and_writes_failed_run_diagnostics(
    tmp_path: Path,
) -> None:
    rejected = _audit(disposition="rejected", reason_code="inference_error")
    staged = stage_dpo_artifacts(
        [],
        [rejected],
        build_summary([rejected]),
        ["no usable samples"],
        _manifest_data(),
        output_name="training.jsonl",
        staging_dir=tmp_path / "stage",
    )
    stats = validate_staged_artifacts(staged, output_name="training.jsonl")
    output = tmp_path / "output"
    with pytest.raises(DpoReportError, match="zero selected"):
        publish_staged_artifacts(
            staged,
            stats,
            output_dir=output,
            output_name="training.jsonl",
            run_id="zero-run",
        )
    assert not output.exists()

    failed = write_failed_run(
        tmp_path / "work",
        "zero-run",
        [rejected],
        build_summary([rejected]),
        ["no usable samples"],
        "zero selected samples",
    )
    assert set(item.name for item in failed.iterdir()) == {
        "audit_records.jsonl",
        "rejected_records.jsonl",
        "summary.json",
        "warnings.log",
        "error.json",
    }
    assert _read_jsonl(failed / "rejected_records.jsonl")[0]["reason_code"] == "inference_error"


def test_compute_artifact_stat_rejects_incomplete_jsonl_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok":true}', encoding="utf-8")
    with pytest.raises(DpoReportError, match="newline"):
        compute_artifact_stat(path, jsonl=True)


def test_compute_artifact_stat_streams_jsonl_without_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"ok":true}\n{"ok":false}\n', encoding="utf-8")
    real_read_bytes = Path.read_bytes

    def reject_whole_file_read(candidate: Path) -> bytes:
        if candidate == path:
            raise AssertionError("JSONL validation must stream")
        return real_read_bytes(candidate)

    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)

    stat = compute_artifact_stat(path, jsonl=True)

    assert stat.line_count == 2
