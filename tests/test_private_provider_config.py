import json

import pytest

from eval_tool import _private_model_router as router


def test_default_private_provider_config_is_industrial_only():
    config = json.loads(router.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    assert "mineru" not in config
    providers = config["vlm_pool"]["providers"]
    assert [provider["name"] for provider in providers] == [
        "qwen3.6-27b",
        "qwen3.6-27b_v2",
    ]
    assert [provider["url"] for provider in providers] == [
        "http://192.168.48.7:18180/v1/chat/completions",
        "http://192.168.48.7:18181/v1/chat/completions",
    ]
    assert {provider["model"] for provider in providers} == {"qwen3.6-27b"}
    assert {provider["api_key_env"] for provider in providers} == {
        "QWEN_36_27B_API_KEY"
    }
    assert {tuple(provider["task_types"]) for provider in providers} == {
        ("industrial_review",)
    }


def test_load_config_rejects_duplicate_provider_names(tmp_path, monkeypatch):
    path = tmp_path / "providers.json"
    path.write_text(
        json.dumps(
            {
                "vlm_pool": {
                    "providers": [
                        {"name": "duplicate", "model": "one"},
                        {"name": "duplicate", "model": "two"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PRIVATE_MODEL_ROUTER_CONFIG", str(path))

    with pytest.raises(router.RouterError, match="duplicate provider name"):
        router.load_config()
