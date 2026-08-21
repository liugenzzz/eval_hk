from __future__ import annotations

import json
import multiprocessing
import os
from dataclasses import replace
from pathlib import Path

import pytest

from eval_tool.dpo_config import DpoInferConfig
from eval_tool.dpo_input import (
    DpoCandidate,
    DpoTurn,
    ImageRef,
    InputSummary,
    SourceRef,
)
from eval_tool.dpo_multimodal import ImageAsset
from eval_tool.judge import JudgeSettings


def _validate_value_record(record):
    if set(record) != {"sample_id", "value"}:
        raise ValueError("record requires exactly sample_id and value")
    if not isinstance(record["sample_id"], str) or not record["sample_id"]:
        raise ValueError("sample_id must be a non-empty string")
    if not isinstance(record["value"], str):
        raise ValueError("value must be a string")


def _store(tmp_path, expected=("a", "b", "c"), **kwargs):
    from eval_tool.dpo_cache import StrictJsonlShardStore

    return StrictJsonlShardStore(
        tmp_path / "inference",
        "fingerprint-a",
        ("sample_id",),
        tuple((sample_id,) for sample_id in expected),
        _validate_value_record,
        **kwargs,
    )


def _record(sample_id, value=None):
    return {"sample_id": sample_id, "value": value or sample_id.upper()}


def test_store_loads_all_worker_shards_and_orders_by_expected_ids(tmp_path):
    store = _store(tmp_path)
    store.append_batch([_record("c")], worker_id=3)
    store.append_batch([_record("a")], worker_id=0)

    loaded = store.load()

    assert list(loaded) == [("a",), ("c",)]
    assert loaded[("c",)]["value"] == "C"


def test_store_streams_shards_without_reading_each_file_into_one_bytes_object(
    tmp_path, monkeypatch
):
    store = _store(tmp_path, expected=("a", "b"))
    store.append_batch([_record("a"), _record("b")])
    store.sync()
    shard = store.shard_path(0)
    real_read_bytes = Path.read_bytes

    def reject_whole_shard_read(path):
        if path == shard:
            raise AssertionError("cache shard must be parsed as a stream")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_whole_shard_read)

    assert list(store.load()) == [("a",), ("b",)]


def test_store_resumes_only_missing_sample_ids(tmp_path):
    store = _store(tmp_path)
    store.append_batch([_record("b")], worker_id=1)

    assert store.pending_keys() == (("a",), ("c",))
    assert store.pending_sample_ids() == ("a", "c")


def test_store_repairs_and_physically_truncates_only_final_partial_line(tmp_path):
    store = _store(tmp_path)
    store.append_batch([_record("a")])
    store.sync()
    shard = store.shard_path(0)
    with shard.open("ab") as stream:
        stream.write(b'{"sample_id":"b","value":')

    assert store.load() == {("a",): _record("a")}
    assert shard.read_bytes().endswith(b"\n")
    assert b'"sample_id":"b"' not in shard.read_bytes()
    assert store.warnings and "truncated tail" in store.warnings[0]


def test_store_repairs_utf8_multibyte_truncation_only_at_the_binary_tail(tmp_path):
    store = _store(tmp_path, expected=("a", "b"))
    store.append_batch([_record("a", "\u5b8c\u6574")])
    store.sync()
    shard = store.shard_path(0)
    tail = json.dumps(
        _record("b", "\u672a\u5b8c"), ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    with shard.open("ab") as stream:
        stream.write(tail[:-1])

    assert list(store.load()) == [("a",)]
    assert shard.read_bytes().endswith(b"\n")


def test_store_rejects_middle_corruption(tmp_path):
    from eval_tool.dpo_cache import DpoCacheError

    store = _store(tmp_path)
    shard = store.shard_path(0)
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_bytes(b'{"sample_id":"a","value":"A"}\n{bad}\n')

    with pytest.raises(DpoCacheError, match="invalid JSON"):
        store.load()
    assert shard.read_bytes().endswith(b"{bad}\n")


def test_store_rejects_duplicate_json_object_keys(tmp_path):
    from eval_tool.dpo_cache import DpoCacheError

    store = _store(tmp_path, expected=("a",))
    shard = store.shard_path(0)
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(
        '{"sample_id":"a","sample_id":"a","value":"A"}\n',
        encoding="utf-8",
    )

    with pytest.raises(DpoCacheError, match="duplicate JSON object key"):
        store.load()


def test_store_rejects_duplicate_or_conflicting_keys_across_shards(tmp_path):
    from eval_tool.dpo_cache import DpoCacheError

    store = _store(tmp_path)
    store.append_batch([_record("a", "first")], worker_id=0)
    store.sync()
    duplicate = store.shard_path(1)
    duplicate.write_text(
        json.dumps(_record("a", "second"), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(DpoCacheError, match="duplicate key"):
        store.load()


def test_store_rejects_unknown_sample_id_and_invalid_schema(tmp_path):
    from eval_tool.dpo_cache import DpoCacheError

    unknown = _store(tmp_path / "unknown", expected=("a",))
    unknown.shard_path(0).parent.mkdir(parents=True, exist_ok=True)
    unknown.shard_path(0).write_text(
        '{"sample_id":"z","value":"Z"}\n', encoding="utf-8"
    )
    with pytest.raises(DpoCacheError, match="unknown key"):
        unknown.load()

    invalid = _store(tmp_path / "invalid", expected=("a",))
    invalid.shard_path(0).parent.mkdir(parents=True, exist_ok=True)
    invalid.shard_path(0).write_text(
        '{"sample_id":"a","value":3}\n', encoding="utf-8"
    )
    with pytest.raises(DpoCacheError, match="value must be a string"):
        invalid.load()


def test_append_flushes_and_bounded_fsync_syncs(tmp_path, monkeypatch):
    import eval_tool.dpo_cache as dpo_cache

    store = _store(tmp_path, expected=("a", "b", "c", "d", "e"), fsync_every=2)
    real_fsync = dpo_cache.os.fsync
    calls = []

    def recording_fsync(descriptor):
        calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(dpo_cache.os, "fsync", recording_fsync)
    store.append_batch([_record(key) for key in "abcde"])
    assert len(calls) == 2
    assert len(store.shard_path(0).read_text(encoding="utf-8").splitlines()) == 5

    store.sync()
    assert len(calls) == 3


def _hold_run_lock(lock_path: str, ready, stop):
    from eval_tool.dpo_cache import DpoRunLock

    with DpoRunLock(Path(lock_path)):
        ready.set()
        stop.wait(30)


def test_run_lock_rejects_a_live_second_process_and_releases_after_owner_death(
    tmp_path,
):
    from eval_tool.dpo_cache import DpoCacheError, DpoRunLock

    lock_path = tmp_path / "run.lock"
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    stop = context.Event()
    process = context.Process(
        target=_hold_run_lock, args=(str(lock_path), ready, stop)
    )
    process.start()
    try:
        assert ready.wait(15)
        with pytest.raises(DpoCacheError, match="already running"):
            DpoRunLock(lock_path).acquire()
    finally:
        process.terminate()
        process.join(15)
    assert not process.is_alive()

    with DpoRunLock(lock_path):
        pass


def test_nonowner_cannot_release_the_run_lock(tmp_path):
    from eval_tool.dpo_cache import DpoCacheError, DpoRunLock

    owner = DpoRunLock(tmp_path / "run.lock").acquire()
    try:
        with pytest.raises(DpoCacheError, match="does not own"):
            DpoRunLock(tmp_path / "run.lock").release()
    finally:
        owner.release()


def test_overwrite_switches_to_a_new_attempt_without_in_place_delete(tmp_path):
    from eval_tool.dpo_cache import DpoCacheError, StrictJsonlShardStore

    old = _store(tmp_path, expected=("a",))
    old.append_batch([_record("a")])
    old.sync()
    old_attempt = old.attempt_dir

    with pytest.raises(DpoCacheError, match="overwrite"):
        StrictJsonlShardStore(
            tmp_path / "inference",
            "fingerprint-b",
            ("sample_id",),
            (("a",),),
            _validate_value_record,
        )

    new = StrictJsonlShardStore(
        tmp_path / "inference",
        "fingerprint-b",
        ("sample_id",),
        (("a",),),
        _validate_value_record,
        overwrite=True,
    )
    assert new.attempt_dir != old_attempt
    assert old_attempt.is_dir()
    assert new.load() == {}


def test_interrupted_overwrite_attempt_resumes_the_new_fingerprint(tmp_path):
    from eval_tool.dpo_cache import StrictJsonlShardStore

    old = _store(tmp_path, expected=("a", "b"))
    old.append_batch([_record("a")])
    old.sync()
    new = StrictJsonlShardStore(
        tmp_path / "inference",
        "fingerprint-b",
        ("sample_id",),
        (("a",), ("b",)),
        _validate_value_record,
        overwrite=True,
    )
    new.append_batch([_record("a", "new")])
    new.sync()

    resumed = StrictJsonlShardStore(
        tmp_path / "inference",
        "fingerprint-b",
        ("sample_id",),
        (("a",), ("b",)),
        _validate_value_record,
        overwrite=True,
    )
    assert resumed.attempt_id == new.attempt_id
    assert resumed.pending_keys() == (("b",),)


def test_attempt_state_recovers_at_each_active_pointer_and_complete_marker_crash_point(
    tmp_path,
):
    phase_dir = tmp_path / "inference"
    store = _store(tmp_path, expected=("a",))
    (phase_dir / ".active.json.interrupted.tmp").write_text("{", encoding="utf-8")
    reopened = _store(tmp_path, expected=("a",))
    assert reopened.attempt_id == store.attempt_id

    reopened.append_batch([_record("a")])
    reopened.sync()
    (reopened.attempt_dir / ".complete.json.interrupted.tmp").write_text(
        "{", encoding="utf-8"
    )
    assert not reopened.is_complete()
    reopened.mark_complete()
    assert reopened.is_complete()
    assert _store(tmp_path, expected=("a",)).attempt_id == store.attempt_id


def test_completion_snapshot_digest_uses_expected_key_order(tmp_path):
    store = _store(tmp_path, expected=("a", "b"))
    store.append_batch([_record("a"), _record("b")])
    store.sync()
    loaded = store.load()
    reversed_snapshot = dict(reversed(tuple(loaded.items())))

    store.mark_complete(snapshot=reversed_snapshot)

    assert store.is_complete()


def _identity_fixture(tmp_path):
    source_path = tmp_path / "input.jsonl"
    image_path = tmp_path / "image.png"
    source = SourceRef(
        input_index=0,
        source_path=source_path,
        container_format="jsonl",
        record_index=0,
        line_number=1,
        source_id="source-1",
        raw_digest="raw-a",
    )
    ref = ImageRef(original="image.png", resolved=image_path)
    candidate = DpoCandidate(
        sample_id="sample-a",
        conversations=(DpoTurn(from_="human", value="<image>\nquestion"),),
        chosen="gold",
        images=(ref,),
        source=source,
        source_format="sharegpt",
        turn_index=0,
        turn_count=1,
    )
    summary = InputSummary(
        source_path=source_path,
        sha256="input-sha",
        byte_count=20,
        container_format="jsonl",
        record_count=1,
    )
    asset = ImageAsset(
        ref=ref, mime_type="image/png", sha256="image-sha", byte_count=10
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    config = DpoInferConfig(model_name="qwen", model_path=model_path)
    return summary, candidate, asset, config


def test_infer_fp_covers_order_normalizer_model_checkpoint_and_actual_messages(
    tmp_path, monkeypatch
):
    import eval_tool.dpo_cache as dpo_cache

    summary, candidate, asset, config = _identity_fixture(tmp_path)
    second = replace(candidate, sample_id="sample-b")
    baseline = dpo_cache.build_dpo_inference_fp(
        (summary,), (candidate, second), {asset.ref.resolved: asset}, config, {"id": "a"}
    )
    assert baseline != dpo_cache.build_dpo_inference_fp(
        (summary,), (second, candidate), {asset.ref.resolved: asset}, config, {"id": "a"}
    )
    changed_message = replace(
        candidate,
        conversations=(DpoTurn(from_="human", value="changed"),),
        images=(),
    )
    assert baseline != dpo_cache.build_dpo_inference_fp(
        (summary,),
        (changed_message, second),
        {asset.ref.resolved: asset},
        config,
        {"id": "a"},
    )
    assert baseline != dpo_cache.build_dpo_inference_fp(
        (replace(summary, sha256="changed"),),
        (candidate, second),
        {asset.ref.resolved: asset},
        config,
        {"id": "a"},
    )
    assert baseline != dpo_cache.build_dpo_inference_fp(
        (summary,), (candidate, second), {asset.ref.resolved: asset}, replace(config, model_name="other"), {"id": "a"}
    )
    assert baseline != dpo_cache.build_dpo_inference_fp(
        (summary,), (candidate, second), {asset.ref.resolved: asset}, config, {"id": "b"}
    )
    monkeypatch.setattr(dpo_cache, "DPO_NORMALIZER_VERSION", "changed")
    assert baseline != dpo_cache.build_dpo_inference_fp(
        (summary,), (candidate, second), {asset.ref.resolved: asset}, config, {"id": "a"}
    )


def test_infer_fp_changes_for_image_bytes_thinking_batch_gpu_dtype_or_device(tmp_path):
    from eval_tool.dpo_cache import build_dpo_inference_fp

    summary, candidate, asset, config = _identity_fixture(tmp_path)

    def fingerprint(current_config=config, current_asset=asset):
        return build_dpo_inference_fp(
            (summary,),
            (candidate,),
            {current_asset.ref.resolved: current_asset},
            current_config,
            {"checkpoint": "a"},
        )

    baseline = fingerprint()
    assert baseline != fingerprint(current_asset=replace(asset, sha256="changed"))
    variants = (
        replace(config, enable_thinking=True),
        replace(config, batch_size=2),
        replace(config, gpu_ids=(0,)),
        replace(config, torch_dtype="bfloat16"),
        replace(config, device_map="cuda:0"),
    )
    assert all(fingerprint(variant) != baseline for variant in variants)


def test_infer_fp_is_stable_when_only_workers_per_gpu_changes(tmp_path):
    from eval_tool.dpo_cache import build_dpo_inference_fp

    summary, candidate, asset, config = _identity_fixture(tmp_path)
    arguments = ((summary,), (candidate,), {asset.ref.resolved: asset})
    assert build_dpo_inference_fp(*arguments, config, {"id": "a"}) == (
        build_dpo_inference_fp(
            *arguments, replace(config, workers_per_gpu=9), {"id": "a"}
        )
    )


def test_infer_fp_adds_image_preprocess_identity_only_when_enabled(
    tmp_path, monkeypatch
):
    import eval_tool.dpo_cache as dpo_cache

    summary, candidate, asset, config = _identity_fixture(tmp_path)
    captured = []
    real_build_phase_fp = dpo_cache.build_phase_fp

    def recording_build_phase_fp(phase, common_identity, expected_keys):
        captured.append(common_identity)
        return real_build_phase_fp(phase, common_identity, expected_keys)

    monkeypatch.setattr(dpo_cache, "build_phase_fp", recording_build_phase_fp)
    arguments = ((summary,), (candidate,), {asset.ref.resolved: asset})

    legacy = dpo_cache.build_dpo_inference_fp(
        *arguments,
        config,
        {"id": "a"},
    )
    configured = dpo_cache.build_dpo_inference_fp(
        *arguments,
        replace(
            config,
            image_min_pixels=65_536,
            image_max_pixels=589_824,
        ),
        {"id": "a"},
    )

    legacy_model = {
        "model_name": config.model_name,
        "model_path": str(config.model_path),
        "enable_thinking": config.enable_thinking,
        "max_new_tokens": config.max_new_tokens,
        "batch_size": config.batch_size,
        "torch_dtype": config.torch_dtype,
        "device_map": config.device_map,
        "gpu_ids": list(config.gpu_ids),
    }
    assert captured[0]["model"] == legacy_model
    assert captured[1]["model"] == {
        **legacy_model,
        "image_preprocess": {
            "profile": "llamafactory-0.9.5-qwen-static-v1",
            "min_pixels": 65_536,
            "max_pixels": 589_824,
        },
    }
    assert configured != legacy


def test_infer_fp_validates_programmatic_image_pixel_bounds(tmp_path):
    from eval_tool.dpo_cache import build_dpo_inference_fp

    summary, candidate, asset, config = _identity_fixture(tmp_path)
    arguments = ((summary,), (candidate,), {asset.ref.resolved: asset})

    with pytest.raises(ValueError, match="provided together"):
        build_dpo_inference_fp(
            *arguments,
            replace(config, image_min_pixels=65_536),
            {"id": "a"},
        )
    with pytest.raises(ValueError, match="image_min_pixels.*<=.*image_max_pixels"):
        build_dpo_inference_fp(
            *arguments,
            replace(
                config,
                image_min_pixels=589_824,
                image_max_pixels=65_536,
            ),
            {"id": "a"},
        )


def test_checkpoint_identity_uses_config_hash_and_weight_name_size_mtime_ns(tmp_path):
    from eval_tool.dpo_cache import checkpoint_identity

    model = tmp_path / "model"
    model.mkdir()
    config = model / "config.json"
    weight = model / "model-00001-of-00002.safetensors"
    config.write_text('{"hidden":1}', encoding="utf-8")
    weight.write_bytes(b"weights")
    timestamp = 1_700_000_000_123_456_700
    os.utime(weight, ns=(timestamp, timestamp))

    identity = checkpoint_identity(model)
    by_path = {entry["path"]: entry for entry in identity["files"]}
    assert by_path["config.json"]["sha256"]
    assert by_path[weight.name] == {
        "path": weight.name,
        "kind": "weight_stat",
        "byte_count": 7,
        "mtime_ns": weight.stat().st_mtime_ns,
    }
    first_digest = identity["digest"]
    config.write_text('{"hidden":2}', encoding="utf-8")
    assert checkpoint_identity(model)["digest"] != first_digest


def test_judge_request_fp_covers_api_base_model_temperature_timeout_tokens_rubric_prompt_messages_mime_and_image_bytes(
    tmp_path,
):
    from eval_tool.dpo_cache import build_judge_request_fp

    _, _, asset, _ = _identity_fixture(tmp_path)
    settings = JudgeSettings(
        api_base="http://judge/v1", model="judge", temperature=0.1, timeout=30, max_retries=2
    )
    material = {
        "rubric": "binary",
        "prompt": "prompt-a",
        "messages": [{"role": "user", "content": "question"}],
    }

    def fp(current_settings=settings, current_material=material, current_asset=asset, tokens=1024):
        return build_judge_request_fp(
            "infer-a", current_settings, current_material, (current_asset,), max_tokens=tokens
        )

    baseline = fp()
    setting_variants = (
        replace(settings, api_base="http://other/v1"),
        replace(settings, model="other"),
        replace(settings, temperature=0.2),
        replace(settings, timeout=31),
        replace(settings, max_retries=3),
    )
    assert all(fp(current_settings=value) != baseline for value in setting_variants)
    assert fp(tokens=2048) != baseline
    for key, value in (
        ("rubric", "v4"),
        ("prompt", "prompt-b"),
        ("messages", [{"role": "user", "content": "changed"}]),
    ):
        assert fp(current_material={**material, key: value}) != baseline
    assert fp(current_asset=replace(asset, mime_type="image/jpeg")) != baseline
    assert fp(current_asset=replace(asset, sha256="other-bytes")) != baseline


def test_judge_request_fp_excludes_api_key(tmp_path):
    from eval_tool.dpo_cache import build_judge_request_fp

    _, _, asset, _ = _identity_fixture(tmp_path)
    first = JudgeSettings(api_key="secret-a")
    second = replace(first, api_key="secret-b")
    material = {"rubric": "binary", "prompt": "p", "messages": []}
    assert build_judge_request_fp("infer", first, material, (asset,)) == (
        build_judge_request_fp("infer", second, material, (asset,))
    )


def test_judge_parse_fp_covers_raw_response_schema_parser_and_score_versions(monkeypatch):
    import eval_tool.dpo_cache as dpo_cache

    baseline = dpo_cache.build_judge_parse_fp("request", '{"correct":false}', "binary", False)
    variants = {
        dpo_cache.build_judge_parse_fp("other", '{"correct":false}', "binary", False),
        dpo_cache.build_judge_parse_fp("request", '{"correct":true}', "binary", False),
        dpo_cache.build_judge_parse_fp("request", '{"correct":false}', "v4", False),
        dpo_cache.build_judge_parse_fp("request", '{"correct":false}', "binary", True),
    }
    assert baseline not in variants
    for constant in (
        "DPO_JUDGE_RESPONSE_SCHEMA_VERSION",
        "DPO_JUDGE_PARSER_VERSION",
        "DPO_JUDGE_SCORE_VERSION",
    ):
        monkeypatch.setattr(dpo_cache, constant, "changed")
        assert (
            dpo_cache.build_judge_parse_fp(
                "request", '{"correct":false}', "binary", False
            )
            != baseline
        )
        monkeypatch.undo()
