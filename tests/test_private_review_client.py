import pytest
from eval_tool.private_review_client import PrivateReviewClient, router


def config():
    return {"vlm_pool": {"fallback": {"enabled": True, "max_attempts": 2, "cooldown_seconds": 300},
            "providers": [{"name": n, "model": n, "url": "http://test", "timeout": 240,
                           "capabilities": ["text", "image"], "max_concurrency": 1, "weight": 1} for n in ["one", "two"]]}}


def test_fallback_and_cooldown(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIVATE_MODEL_ROUTER_STATE", str(tmp_path / "state.json"))
    calls = []
    def http(url, payload, headers, timeout):
        calls.append(payload["model"])
        assert timeout == 240
        if payload["model"] == "one": raise router.RouterError("failure")
        return {"choices": [{"message": {"content": "ok"}}]}
    monkeypatch.setattr(router, "http_json", http)
    client = PrivateReviewClient(config=config())
    assert client.review("prompt")["provider"] == "two"
    assert calls == ["one", "two"]
    calls.clear(); client.review("prompt"); assert calls == ["two"]


def test_missing_key_skipped(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIVATE_MODEL_ROUTER_STATE", str(tmp_path / "state.json"))
    monkeypatch.setattr(router, "read_api_key", lambda _: None)
    c = config(); c["vlm_pool"]["providers"][0]["api_key_env"] = "SECRET_VAR"
    monkeypatch.setattr(router, "http_json", lambda *a: {"choices": [{"message": {"content": "ok"}}]})
    assert PrivateReviewClient(config=c).review("prompt")["provider"] == "two"
    with pytest.raises(RuntimeError, match="failed"):
        PrivateReviewClient(provider="one", config=c).review("prompt")


def test_error_text_never_leaks(monkeypatch):
    def fail(*args): raise RuntimeError("secret-value")
    monkeypatch.setattr(router, "run_chat", fail)
    with pytest.raises(RuntimeError) as error: PrivateReviewClient(config=config()).review("p")
    assert "secret-value" not in str(error.value)


def test_image_payload_and_empty_response(tmp_path, monkeypatch):
    monkeypatch.setenv("PRIVATE_MODEL_ROUTER_STATE", str(tmp_path / "state.json"))
    image = tmp_path / "i.png"; image.write_bytes(b"image")
    def http(url, payload, headers, timeout):
        assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
        return {"choices": [{"message": {"content": ""}}]}
    monkeypatch.setattr(router, "http_json", http)
    with pytest.raises(RuntimeError): PrivateReviewClient(config=config()).review("p", [image])


@pytest.mark.parametrize("content", [None, [], {}, 1, "", "   "])
def test_nontext_content_never_passes(monkeypatch, content):
    monkeypatch.setattr(router, "run_chat", lambda *args: {"content": content, "provider": "test", "model": "test"})
    with pytest.raises(RuntimeError, match="non-text or empty"):
        PrivateReviewClient(config=config()).review("p")


def _provider(name, model):
    return {
        "name": name,
        "model": model,
        "url": "http://test/v1/chat/completions",
        "capabilities": ["text", "image"],
        "task_types": ["industrial_review"],
    }


def _routing_config(*providers):
    return {"vlm_pool": {"providers": list(providers)}}


def test_requested_provider_prefers_exact_name_over_shared_model():
    cfg = _routing_config(
        _provider("qwen3.6-27b", "qwen3.6-27b"),
        _provider("qwen3.6-27b_v2", "qwen3.6-27b"),
    )

    matches = router.eligible_providers(
        cfg, "industrial_review", "text", requested="qwen3.6-27b"
    )

    assert [provider["name"] for provider in matches] == ["qwen3.6-27b"]


def test_requested_ambiguous_model_without_same_named_provider_is_rejected():
    cfg = _routing_config(
        _provider("endpoint-a", "shared-model"),
        _provider("endpoint-b", "shared-model"),
    )

    with pytest.raises(router.RouterError, match="ambiguous"):
        router.eligible_providers(
            cfg, "industrial_review", "text", requested="shared-model"
        )


def test_requested_unique_model_falls_back_to_its_provider():
    cfg = _routing_config(
        _provider("endpoint-a", "unique-model"),
        _provider("endpoint-b", "other-model"),
    )

    matches = router.eligible_providers(
        cfg, "industrial_review", "text", requested="unique-model"
    )

    assert [provider["name"] for provider in matches] == ["endpoint-a"]
