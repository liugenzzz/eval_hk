import json
import pytest
from eval_tool.industrial_mapping import scan, freeze, file_sha256
from eval_tool.industrial_review import mapping_review, sample_review, sample_input_hash


class Client:
    cache_settings = {"provider": "mock"}
    def __init__(self, content=None, fail=False): self.calls = 0; self.content = content; self.fail = fail
    def review(self, prompt, image_paths=()):
        self.calls += 1
        if self.fail: raise RuntimeError("secret")
        e = json.loads(prompt.split("INPUT:\n")[1])
        answer = self.content or {"entry_id": e["entry_id"], "decision": "accepted", "canonical": e["canonical_candidates"][0], "defect_status": "abnormal", "reason": "semantic match"}
        return {"content": json.dumps(answer), "provider": "mock", "model": "mock"}


def fixture(tmp_path):
    en = tmp_path / "en.jsonl"; zh = tmp_path / "zh.jsonl"
    for path, answer in [(en, "Category: aircraft; Anomaly type: crack; Description: crack"), (zh, "类别：飞机；异常类型：裂纹；描述：裂纹")]:
        path.write_text(json.dumps({"images": ["/defect-detect-images/domain/i.png"], "conversations": [{"from": "human", "value": "what?"}, {"from": "gpt", "value": answer}]}), encoding="utf-8")
    scan(en, zh, tmp_path / "scan")
    return tmp_path / "scan/candidate.yaml", tmp_path / "scan/review_queue.jsonl", tmp_path / "reviews.jsonl"


def test_resume_and_candidate_binding(tmp_path):
    candidate, queue, out = fixture(tmp_path); c = Client()
    mapping_review(candidate, queue, out, client=c)
    count = c.calls; assert count > 0
    mapping_review(candidate, queue, out, client=c, resume=True); assert c.calls == count
    candidate.write_text(candidate.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="match candidate"):
        mapping_review(candidate, queue, out, client=c, resume=True)


def test_schema_errors_retry_and_never_accept(tmp_path):
    candidate, queue, out = fixture(tmp_path); c = Client({"decision": "accepted"})
    mapping_review(candidate, queue, out, client=c)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert all(r["status"] == "error" and r["decision"] == "uncertain" for r in rows)
    old = c.calls; mapping_review(candidate, queue, out, client=c, resume=True); assert c.calls > old


def test_api_errors_retry_and_redact(tmp_path):
    candidate, queue, out = fixture(tmp_path); c = Client(fail=True)
    mapping_review(candidate, queue, out, client=c)
    assert "secret" not in out.read_text()
    old = c.calls; mapping_review(candidate, queue, out, client=c, resume=True); assert c.calls > old


def test_hash_includes_question_and_mapping():
    assert sample_input_hash({"question": "a"}, "en", "1") != sample_input_hash({"question": "b"}, "en", "1")
    assert sample_input_hash({}, "en", "1") != sample_input_hash({}, "zh", "1")
    assert sample_input_hash({}, "en", "1") != sample_input_hash({}, "en", "2")


def test_sample_image_change_and_resume(tmp_path):
    candidate, queue, out = fixture(tmp_path)
    mapping_review(candidate, queue, out, client=Client())
    frozen = tmp_path / "frozen.yaml"; freeze(candidate, out, frozen)
    image_root = tmp_path / "images"; (image_root / "domain").mkdir(parents=True)
    image = image_root / "domain/i.png"; image.write_bytes(b"one")
    class SampleClient:
        cache_settings = {"provider": "mock"}
        calls = 0
        def review(self, prompt, image_paths=()):
            self.calls += 1; assert image_paths
            return {"content": '{"decision":"usable","reason":"visually grounded"}', "provider":"mock", "model":"mock"}
    client = SampleClient(); target = tmp_path / "samples.jsonl"
    kwargs = dict(zh=tmp_path / "zh.jsonl", en=tmp_path / "en.jsonl", mapping=frozen, out=target, client=client, image_root=image_root)
    sample_review(**kwargs); assert client.calls == 2
    sample_review(**kwargs, resume=True); assert client.calls == 2
    image.write_bytes(b"two")
    sample_review(**kwargs, resume=True); assert client.calls == 4
    rows = [json.loads(x) for x in target.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["image_sha256"] != rows[-1]["image_sha256"]


def test_sample_missing_image_never_usable(tmp_path):
    candidate, queue, out = fixture(tmp_path)
    mapping_review(candidate, queue, out, client=Client())
    frozen = tmp_path / "frozen.yaml"; freeze(candidate, out, frozen)
    target = tmp_path / "samples.jsonl"; client = Client()
    sample_review(tmp_path / "zh.jsonl", tmp_path / "en.jsonl", frozen, target, image_root=tmp_path / "absent", client=client)
    assert client.calls == 0
    rows = [json.loads(x) for x in target.read_text(encoding="utf-8").splitlines()]
    assert all(r["status"] == "error" and r["decision"] != "usable" for r in rows)


def test_sample_review_silently_skips_review_excluded_categories(tmp_path):
    candidate, queue, out = fixture(tmp_path)
    mapping_review(candidate, queue, out, client=Client(fail=True))
    frozen = tmp_path / "frozen.yaml"; freeze(candidate, out, frozen)
    target = tmp_path / "samples.jsonl"; client = Client()
    result = sample_review(tmp_path / "zh.jsonl", tmp_path / "en.jsonl",
                           frozen, target, client=client)
    assert client.calls == 0
    assert result["skipped_excluded"] == 2
    assert target.read_text(encoding="utf-8") == ""

