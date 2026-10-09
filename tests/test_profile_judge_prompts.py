"""Each built-in profile's judge rubric must be written for that profile's content.

A rubric framed for another domain misjudges: the equipment rubric failed correct
answers about a formula on a book page as "not about military equipment", and the
aviation image rubric, given a video row with no frames, judged park answers against
an imagined "aircraft cockpit".
"""

import json
from pathlib import Path

import pytest

from eval_tool.config import load_pipeline_config
from eval_tool.judge import JudgeClient
from eval_tool.rubrics import apply_rubric

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ROOT / "eval_tool" / "profiles"

# prompt family -> profiles allowed to use it
DOMAIN_PROMPTS = {
    "judge_equip_": {"equip_zh_v1"},
    "judge_vqa_": {"aircraft_maintenance_v2"},
    "judge_industrial_": {"industrial_bilingual_v1", "industrial_mixed_v1"},
}


def judge_prompts(profile):
    raw = json.loads((PROFILES / f"{profile}.json").read_text(encoding="utf-8"))
    return dict((raw.get("judge") or {}).get("prompt_files") or {})


@pytest.mark.parametrize("profile", sorted(p.stem for p in PROFILES.glob("*.json")))
def test_domain_framed_rubrics_stay_in_their_domain(profile):
    for path in judge_prompts(profile).values():
        for prefix, allowed in DOMAIN_PROMPTS.items():
            if Path(path).name.startswith(prefix):
                assert profile in allowed, f"{profile} uses {path}"


def test_book_and_park_use_their_own_rubrics():
    assert judge_prompts("book_zh_v1") == {
        "pointwise": "prompts/judge_doc_pointwise_v2.txt",
        "pairwise": "prompts/judge_doc_pairwise_v1.txt",
    }
    assert judge_prompts("park_video_v1") == {
        "pointwise": "prompts/judge_video_ref_pointwise_v1.txt",
        "pairwise": "prompts/judge_video_ref_pairwise_v1.txt",
    }


def test_video_rubric_tells_the_judge_it_gets_no_frames():
    for name in ("judge_video_ref_pointwise_v1.txt", "judge_video_ref_pairwise_v1.txt"):
        text = (ROOT / "prompts" / name).read_text(encoding="utf-8")
        assert "看不到视频" in text and "参考答案" in text


@pytest.mark.parametrize("reply, hit", [
    ('{"correct": true, "reason": "与参考答案一致"}', 1),
    ('{"correct": false, "reason": "与参考答案矛盾"}', 0),
])
def test_new_rubric_replies_parse(monkeypatch, reply, hit):
    config = load_pipeline_config(ROOT / "book.example.json").to_eval_config()
    for version in ("doc_v1", "video_ref_v1"):
        judge = apply_rubric(config, version).judge
        monkeypatch.setattr(JudgeClient, "_post", lambda *a, **kw: reply)
        verdict = JudgeClient(judge).judge_pointwise("question", "reference", "prediction")
        assert verdict["hit"] == hit
        assert verdict["quality_score"] == 100.0 * hit


def doc_v2_reply(**fields):
    reply = {"rubric": "doc_v2", "on_topic": True, "fact_score": 100, "param_score": 80,
             "visual_score": None, "fabrication_score": 100, "style_score": 90,
             "reasoning": "基金编号一位数字转录错误"}
    reply.update(fields)
    return json.dumps(reply, ensure_ascii=False)


@pytest.mark.parametrize("reply, quality, hit, failure", [
    # one mistyped digit in a long extraction costs points, not the whole answer
    (doc_v2_reply(), 92.0, 1, None),
    # the gate caps an answer to another question, whatever its axes say
    (doc_v2_reply(on_topic=False), 30.0, 0, "off_topic"),
    # fabrication halves accuracy at worst
    (doc_v2_reply(fabrication_score=0), 46.0, 0, None),
    # the judge dropping the tag still reads as doc_v2
    (doc_v2_reply(rubric=None), 92.0, 1, None),
])
def test_doc_v2_is_graded_with_a_topic_gate(monkeypatch, reply, quality, hit, failure):
    config = load_pipeline_config(ROOT / "book.example.json").to_eval_config()
    monkeypatch.setattr(JudgeClient, "_post", lambda *a, **kw: reply)
    verdict = JudgeClient(config.judge).judge_pointwise("question", "reference", "prediction")
    assert verdict["quality_score"] == quality
    assert verdict["hit"] == hit
    assert verdict["failure_type"] == failure
    assert verdict["equipment_correct"] is None
