import json

import pytest

from eval_tool.park_abnormal import (
    TAXONOMY_VERSION,
    load_annotations,
    match_event_types,
    score_predictions,
    temporal_iou,
    validate_annotation,
)


def reference(**changes):
    row = {
        "taxonomy_version": TAXONOMY_VERSION,
        "clip_id": "nwpu-test-001-10-15",
        "video": "/data/nwpu/test/001.mp4",
        "duration_s": 30,
        "event_type": "running",
        "interval": {"start_s": 10, "end_s": 15},
        "description": "A person runs across the campus walkway.",
        "reviews": [
            {"reviewer": "r1", "event_type": "running", "interval": {"start_s": 10, "end_s": 15}, "description": "A person runs on a walkway."},
            {"reviewer": "r2", "event_type": "running", "interval": {"start_s": 10.2, "end_s": 15.1}, "description": "Someone runs outside."},
        ],
        "review_agreement": {"event_type": True, "interval": True, "description": True},
        "verification": "verified",
        "adjudicator": "r3",
    }
    row.update(changes)
    return row


def test_verified_annotation_and_independent_scores():
    refs = {"nwpu-test-001-10-15": reference()}
    predictions = [{
        "clip_id": "nwpu-test-001-10-15", "answer": "有人在跑步。",
        "interval": {"start_s": 12, "end_s": 17},
    }]
    result = score_predictions(refs, predictions, require_time=True,
                               description_judge=lambda ref, answer: 0.75)[0]
    assert result["type_hit"] is True
    assert result["temporal_iou"] == pytest.approx(3 / 7)
    assert result["description_score"] == 0.75


def test_agreed_reviews_do_not_require_a_third_reviewer():
    row = reference()
    row.pop("adjudicator")
    assert validate_annotation(row)["verification"] == "verified"


def test_no_semantic_or_temporal_score_when_not_requested():
    row = score_predictions({"x": reference(clip_id="x")},
                            [{"clip_id": "x", "answer": "running"}])[0]
    assert row["temporal_iou"] is None
    assert row["description_score"] is None


@pytest.mark.parametrize("change", [
    {"verification": "draft"},
    {"reviews": []},
    {"reviews": [{"reviewer": "same", "event_type": "running", "interval": {"start_s": 10, "end_s": 15}, "description": "Runs"}] * 2},
    {"event_type": "unknown"},
    {"interval": {"start_s": 5, "end_s": 5}},
    {"interval": {"start_s": float("nan"), "end_s": 5}},
    {"duration_s": 12},
])
def test_invalid_or_unverified_annotations_rejected(change):
    with pytest.raises(ValueError):
        validate_annotation(reference(**change))


def test_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "annotations.jsonl"
    path.write_text(json.dumps(reference()) + "\n" + json.dumps(reference()) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate clip_id"):
        load_annotations(path)


def test_disagreement_requires_explicit_adjudication():
    row = reference()
    row["reviews"][1] = {"reviewer": "r2", "event_type": "fighting", "interval": {"start_s": 11, "end_s": 16}, "description": "Two people fight."}
    with pytest.raises(ValueError, match="agreement conflicts"):
        validate_annotation(row)
    row["review_agreement"] = {"event_type": False, "interval": False, "description": False}
    with pytest.raises(ValueError, match="adjudication_reason"):
        validate_annotation(row)
    row["adjudication_reason"] = "Reviewed frames 300-450; one person runs, no contact."
    assert validate_annotation(row)["event_type"] == "running"
    row["adjudicator"] = "r1"
    with pytest.raises(ValueError, match="adjudicator must differ"):
        validate_annotation(row)


def test_predictions_require_same_ids_and_time_when_asked():
    refs = {"x": reference(clip_id="x")}
    with pytest.raises(ValueError, match="clip IDs differ"):
        score_predictions(refs, [{"clip_id": "y", "answer": "running"}])
    with pytest.raises(ValueError, match="predicted interval required"):
        score_predictions(refs, [{"clip_id": "x", "answer": "running"}], require_time=True)


def test_ambiguous_type_answer_does_not_get_credit():
    assert match_event_types("有人奔跑并打架") == {"running", "fighting"}
    assert match_event_types("没有奔跑，也没有打架") == set()
    assert match_event_types("not running") == set()
    row = score_predictions({"x": reference(clip_id="x")},
                            [{"clip_id": "x", "answer": "有人奔跑并打架"}])[0]
    assert row["type_hit"] is False
    assert temporal_iou({"start_s": 0, "end_s": 2}, {"start_s": 3, "end_s": 4}) == 0


def test_structured_type_cannot_rescue_answer_without_event_recognition():
    row = score_predictions({"x": reference(clip_id="x")},
                            [{"clip_id": "x", "answer": "Nothing unusual happened.",
                              "event_type": "running"}])[0]
    assert row["type_hit"] is False
    assert row["structured_type_hit"] is True


def test_false_agreement_must_match_independent_reviews():
    row = reference()
    row["review_agreement"] = {"event_type": False, "interval": True, "description": True}
    row["adjudication_reason"] = "Rechecked the clip."
    with pytest.raises(ValueError, match="event_type agreement conflicts"):
        validate_annotation(row)
