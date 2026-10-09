"""The judge runs with thinking off, and never parses the reasoning if it gets one.

Qwen3.x behind vLLM thinks by default; the reasoning often already contains a
half-made verdict ({"score": …}). When thinking uses up max_tokens, the last complete
JSON object is one from the reasoning, and it would be read as the verdict.
"""

import io
import json

from eval_tool.judge import JudgeClient, JudgeSettings


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _capture(monkeypatch, content):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["body"] = json.loads(req.data.decode("utf-8"))
        return _Response(json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return sent


def test_default_keeps_the_primary_fingerprint_explicit_thinking_gets_its_own():
    # the primary judge fingerprint is pinned (tests/test_cross_check_judges.py) so the
    # verdict caches built up so far stay valid
    off, unset, on = (JudgeSettings(enable_thinking=v).fingerprint for v in (False, None, True))
    assert off == unset != on


def test_thinking_is_switched_off_in_the_request(monkeypatch):
    sent = _capture(monkeypatch, '{"score": 8}')
    JudgeClient(JudgeSettings()).judge_messages([{"role": "user", "content": "q"}])
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_null_setting_leaves_the_field_out(monkeypatch):
    sent = _capture(monkeypatch, '{"score": 8}')
    JudgeClient(JudgeSettings(enable_thinking=None)).judge_messages([{"role": "user", "content": "q"}])
    assert "chat_template_kwargs" not in sent["body"]


def test_reasoning_before_think_close_is_dropped(monkeypatch):
    _capture(monkeypatch, '先想想，可以给 {"score": 3}，不对，应该更高。\n</think>\n\n{"score": 8}')
    reply = JudgeClient(JudgeSettings()).judge_messages([{"role": "user", "content": "q"}])
    assert reply == '{"score": 8}'

