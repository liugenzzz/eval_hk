from eval_tool.cache import JsonlCache


def test_jsonl_cache_persists_and_reloads_records(tmp_path):
    cache_path = tmp_path / "judge_cache.jsonl"
    cache = JsonlCache(cache_path, key_fields=("model", "dataset", "index"))

    cache.set({"model": "base", "dataset": "vqa", "index": "42"}, {"hit": 1, "reason": "ok"})

    reloaded = JsonlCache(cache_path, key_fields=("model", "dataset", "index"))
    assert reloaded.get({"model": "base", "dataset": "vqa", "index": "42"}) == {
        "hit": 1,
        "reason": "ok",
    }


def test_jsonl_cache_overwrites_in_memory_with_latest_record(tmp_path):
    cache_path = tmp_path / "judge_cache.jsonl"
    cache = JsonlCache(cache_path, key_fields=("model", "dataset", "index"))

    key = {"model": "base", "dataset": "vqa", "index": "42"}
    cache.set(key, {"hit": 0, "reason": "old"})
    cache.set(key, {"hit": 1, "reason": "new"})

    reloaded = JsonlCache(cache_path, key_fields=("model", "dataset", "index"))
    assert reloaded.get(key) == {"hit": 1, "reason": "new"}
