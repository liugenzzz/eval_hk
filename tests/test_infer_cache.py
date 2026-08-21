import importlib
import hashlib
import json

import pytest

from eval_tool.imaging import IMAGE_PREPROCESS_PROFILE


def _legacy_fingerprint(model_path, prompt_text, max_new_tokens, dataset_key, dataset_name, rows):
    input_digest = hashlib.sha256()
    for row in rows:
        input_digest.update(
            json.dumps(
                row,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        input_digest.update(b"\n")
    payload = {
        "model_path": str(model_path.resolve()),
        "prompt_text": prompt_text,
        "max_new_tokens": max_new_tokens,
        "dataset_key": dataset_key,
        "dataset_name": dataset_name,
        "input_digest": input_digest.hexdigest(),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def test_fingerprint_is_stable_and_changes_with_inference_input(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    rows = [
        {
            "index": "1",
            "question": "q",
            "history": "",
            "image": "img",
            "A": "first option",
        }
    ]

    first = infer_cache.build_infer_fingerprint(
        tmp_path / "model", "prompt", 32, "vqa", "aero_vqa", rows
    )
    repeated = infer_cache.build_infer_fingerprint(
        tmp_path / "model", "prompt", 32, "vqa", "aero_vqa", rows
    )
    changed = infer_cache.build_infer_fingerprint(
        tmp_path / "model",
        "prompt",
        32,
        "vqa",
        "aero_vqa",
        [{**rows[0], "A": "changed option"}],
    )

    assert first == repeated
    assert first != changed
    assert len(first) == 16


def test_fingerprint_covers_every_run_identity_field(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    rows = [{"index": "1", "question": "q"}]
    baseline = infer_cache.build_infer_fingerprint(
        tmp_path / "model", "prompt", 32, "vqa", "aero_vqa", rows
    )
    variants = {
        infer_cache.build_infer_fingerprint(
            tmp_path / "other-model", "prompt", 32, "vqa", "aero_vqa", rows
        ),
        infer_cache.build_infer_fingerprint(
            tmp_path / "model", "changed", 32, "vqa", "aero_vqa", rows
        ),
        infer_cache.build_infer_fingerprint(
            tmp_path / "model", "prompt", 64, "vqa", "aero_vqa", rows
        ),
        infer_cache.build_infer_fingerprint(
            tmp_path / "model", "prompt", 32, "mcq", "aero_vqa", rows
        ),
        infer_cache.build_infer_fingerprint(
            tmp_path / "model", "prompt", 32, "vqa", "other_vqa", rows
        ),
    }

    assert baseline not in variants
    assert len(variants) == 5


def test_fingerprint_changes_for_each_configured_image_pixel_bound(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    args = (tmp_path / "model", "prompt", 32, "vqa", "aero_vqa", [{"index": "1"}])

    fingerprints = {
        infer_cache.build_infer_fingerprint(
            *args, image_min_pixels=65536, image_max_pixels=589824
        ),
        infer_cache.build_infer_fingerprint(
            *args, image_min_pixels=32768, image_max_pixels=589824
        ),
        infer_cache.build_infer_fingerprint(
            *args, image_min_pixels=65536, image_max_pixels=786432
        ),
    }

    assert len(fingerprints) == 3


def test_fingerprint_payload_identifies_image_preprocess_profile(tmp_path, monkeypatch):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    original_dumps = infer_cache.json.dumps
    payloads = []

    def capture_payload(value, *args, **kwargs):
        if isinstance(value, dict) and "input_digest" in value:
            payloads.append(value)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(infer_cache.json, "dumps", capture_payload)
    infer_cache.build_infer_fingerprint(
        tmp_path / "model",
        "prompt",
        32,
        "vqa",
        "aero_vqa",
        [{"index": "1"}],
        image_min_pixels=65536,
        image_max_pixels=589824,
    )

    assert payloads[-1]["image_preprocess"] == {
        "profile": IMAGE_PREPROCESS_PROFILE,
        "min_pixels": 65536,
        "max_pixels": 589824,
    }


def test_none_image_pixel_bounds_preserve_legacy_fingerprint_payload(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    model_path = tmp_path / "model"
    rows = [{"index": "1", "question": "q"}]
    expected = _legacy_fingerprint(
        model_path, "prompt", 32, "vqa", "aero_vqa", rows
    )

    omitted = infer_cache.build_infer_fingerprint(
        model_path, "prompt", 32, "vqa", "aero_vqa", rows
    )
    explicit_none = infer_cache.build_infer_fingerprint(
        model_path,
        "prompt",
        32,
        "vqa",
        "aero_vqa",
        rows,
        image_min_pixels=None,
        image_max_pixels=None,
    )

    assert omitted == expected
    assert explicit_none == expected


@pytest.mark.parametrize(
    ("image_min_pixels", "image_max_pixels", "message"),
    [
        (65536, None, "provided together"),
        (True, 589824, "image_min_pixels.*int"),
        (589825, 589824, "image_min_pixels.*<=.*image_max_pixels"),
    ],
)
def test_fingerprint_defensively_validates_image_pixel_bounds(
    tmp_path, image_min_pixels, image_max_pixels, message
):
    infer_cache = importlib.import_module("eval_tool.infer_cache")

    with pytest.raises(ValueError, match=message):
        infer_cache.build_infer_fingerprint(
            tmp_path / "model",
            "prompt",
            32,
            "vqa",
            "aero_vqa",
            [],
            image_min_pixels=image_min_pixels,
            image_max_pixels=image_max_pixels,
        )


def test_store_reloads_all_worker_shards(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    store = infer_cache.InferShardStore(tmp_path, "vqa", "abc")
    store.append_batch([("1", "a")], worker_id=0)
    store.append_batch([("2", "b")], worker_id=3)

    reloaded = infer_cache.InferShardStore(tmp_path, "vqa", "abc")

    assert reloaded.load() == {"1": "a", "2": "b"}
    assert reloaded.shard_path(0).name == "vqa__abc__w0.jsonl"


def test_store_rejects_duplicate_index_across_shards(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    store = infer_cache.InferShardStore(tmp_path, "vqa", "abc")
    store.append_batch([("1", "a")], worker_id=0)
    store.append_batch([("1", "a")], worker_id=1)

    with pytest.raises(infer_cache.InferCacheError, match="duplicate index '1'"):
        store.load()


def test_store_ignores_and_reports_truncated_last_line(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    store = infer_cache.InferShardStore(tmp_path, "vqa", "abc")
    store.append_batch([("1", "complete")])
    with store.shard_path().open("a", encoding="utf-8") as stream:
        stream.write('{"index":')

    assert store.load() == {"1": "complete"}
    assert len(store.warnings) == 1
    assert "truncated JSON" in store.warnings[0]

    store.append_batch([("2", "resumed")])

    assert store.load() == {"1": "complete", "2": "resumed"}


def test_store_rejects_records_missing_required_fields(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    store = infer_cache.InferShardStore(tmp_path, "vqa", "abc")
    store.shard_path().parent.mkdir(parents=True, exist_ok=True)
    store.shard_path().write_text('{"index":"1"}\n', encoding="utf-8")

    with pytest.raises(infer_cache.InferCacheError, match="requires string index and prediction"):
        store.load()


@pytest.mark.parametrize(
    "records,message",
    [
        ([("", "answer")], "non-empty index"),
        ([("1", "a"), ("1", "b")], "duplicate index '1' in batch"),
    ],
)
def test_store_rejects_invalid_batch_before_writing(tmp_path, records, message):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    store = infer_cache.InferShardStore(tmp_path, "vqa", "abc")

    with pytest.raises(infer_cache.InferCacheError, match=message):
        store.append_batch(records)

    assert not store.shard_path().exists()


def test_store_reports_foreign_fingerprints_and_cleans_only_requested_dataset(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    current = infer_cache.InferShardStore(tmp_path, "vqa", "abc")
    foreign = infer_cache.InferShardStore(tmp_path, "vqa", "def")
    other_dataset = infer_cache.InferShardStore(tmp_path, "mcq", "zzz")
    current.append_batch([("1", "current")], worker_id=0)
    foreign.append_batch([("1", "foreign")])
    other_dataset.append_batch([("1", "other")])

    assert current.foreign_fingerprints() == {"def"}

    current.clear_current()
    assert not current.shard_path(0).exists()
    assert foreign.shard_path().exists()

    current.clean_dataset()
    assert not foreign.shard_path().exists()
    assert other_dataset.shard_path().exists()


def test_manifest_round_trips_through_atomic_json_write(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    path = tmp_path / "base_aero_vqa.infer.json"
    manifest = infer_cache.InferManifest(
        fingerprint="abc",
        dataset_name="aero_vqa",
        row_count=2,
        index_digest=infer_cache.build_index_digest(["1", "2"]),
    )

    infer_cache.write_manifest(path, manifest)

    assert infer_cache.read_manifest(path) == manifest
    assert not path.with_name(f"{path.name}.tmp").exists()


def test_index_digest_distinguishes_embedded_newlines_from_multiple_indexes():
    infer_cache = importlib.import_module("eval_tool.infer_cache")

    assert infer_cache.build_index_digest(["a\nb"]) != infer_cache.build_index_digest(
        ["a", "b"]
    )


def test_inference_lock_rejects_second_owner_and_releases(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    path = tmp_path / "vqa.lock"

    with infer_cache.InferenceLock(path):
        assert path.is_file()
        with pytest.raises(infer_cache.InferCacheError, match="inference already running"):
            with infer_cache.InferenceLock(path):
                raise AssertionError("second lock must not be acquired")

    assert not path.exists()


def test_inference_lock_removes_file_when_initialization_fails(tmp_path, monkeypatch):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    path = tmp_path / "vqa.lock"

    def fail_fsync(descriptor):
        raise OSError("disk failure")

    monkeypatch.setattr(infer_cache.os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="disk failure"):
        with infer_cache.InferenceLock(path):
            raise AssertionError("lock body must not run")

    assert not path.exists()


def test_read_manifest_rejects_wrong_field_types(tmp_path):
    infer_cache = importlib.import_module("eval_tool.infer_cache")
    path = tmp_path / "invalid.infer.json"
    path.write_text(
        '{"fingerprint":{},"dataset_name":"vqa","row_count":2,"index_digest":"x"}',
        encoding="utf-8",
    )

    with pytest.raises(infer_cache.InferCacheError, match="invalid inference manifest"):
        infer_cache.read_manifest(path)
