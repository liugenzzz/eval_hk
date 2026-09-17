import importlib.util
import json
from pathlib import Path

import yaml

from eval_tool.industrial_mapping import file_sha256


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "probe_industrial_review.py"
SPEC = importlib.util.spec_from_file_location("probe_industrial_review", SCRIPT)
probe_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe_module)


class Client:
    def __init__(self, provider=None):
        self.provider = provider
        self.config = {"vlm_pool": {"providers": [{"name": provider, "api_key_env": "PROBE_TEST_KEY"}]}}


def build_fixture(tmp_path):
    scan = tmp_path / "mapping_scan"
    scan.mkdir(parents=True)
    entry = {
        "id": "entry-1",
        "kind": "defect",
        "source_domain": "weld",
        "language": "zh",
        "alias": "裂纹",
        "object_category": "weld",
        "canonical": "crack",
        "defect_status": "unknown",
        "status": "uncertain",
        "decision_source": "shared_image",
        "canonical_candidates": ["crack"],
        "evidence": {},
    }
    candidate = scan / "candidate.yaml"
    candidate.write_text(yaml.safe_dump({"status": "candidate", "entries": [entry]}), encoding="utf-8")
    queued = dict(entry, entry_id=entry["id"], candidate_sha256=file_sha256(candidate))
    (scan / "review_queue.jsonl").write_text(json.dumps(queued, ensure_ascii=False) + "\n", encoding="utf-8")
    return candidate, queued


def test_probe_success(tmp_path):
    _, queued = build_fixture(tmp_path)

    def runner(args, config):
        assert queued["entry_id"] in args.prompt
        content = {"entry_id": "entry-1", "decision": "accepted", "canonical": "crack", "defect_status": "abnormal", "reason": "same meaning"}
        return {"provider": "qwen3.6-27b_v2", "model": "qwen3.6-27b", "content": json.dumps(content)}

    result = probe_module.probe(tmp_path, "qwen3.6-27b_v2", runner=runner, client_factory=Client)
    assert result["ok"] is True
    assert result["schema_valid"] is True
    assert result["decision"]["canonical"] == "crack"


def test_probe_http_error_is_redacted(tmp_path, monkeypatch):
    build_fixture(tmp_path)
    monkeypatch.setenv("PROBE_TEST_KEY", "super-secret-key")

    def runner(args, config):
        raise probe_module.router.RouterError("HTTP 400: Bearer super-secret-key body=bad request")

    result = probe_module.probe(tmp_path, "qwen3.6-27b_v2", runner=runner, client_factory=Client)
    assert result["ok"] is False
    assert "HTTP 400" in result["error"]
    assert "bad request" in result["error"]
    assert "super-secret-key" not in json.dumps(result)


def test_probe_rejects_queue_candidate_mismatch(tmp_path):
    candidate, _ = build_fixture(tmp_path)
    candidate.write_text(candidate.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    result = probe_module.probe(tmp_path, "qwen3.6-27b_v2", runner=lambda *_: None, client_factory=Client)
    assert result["ok"] is False
    assert "queue does not match candidate" in result["error"]


def test_probe_rejects_empty_queue(tmp_path):
    build_fixture(tmp_path)
    (tmp_path / "mapping_scan" / "review_queue.jsonl").write_text("", encoding="utf-8")
    result = probe_module.probe(tmp_path, "qwen3.6-27b_v2", runner=lambda *_: None, client_factory=Client)
    assert result["ok"] is False
    assert "review queue is empty" in result["error"]
