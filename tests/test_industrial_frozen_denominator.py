import pandas as pd
import pytest

from eval_tool.scorers.industrial import validate_frozen_reference_scores


def test_frozen_reference_requires_usable_scored_judge_result():
    frame = pd.DataFrame([
        {"index": "zh_a__t1", "meta.reference_validity": "usable", "quality_score": 80,
         "industrial_reference_status": "usable"},
        {"index": "en_a__t1", "meta.reference_validity": "usable", "quality_score": None,
         "industrial_reference_status": "conflict"},
    ])
    with pytest.raises(ValueError, match="en_a__t1.*retry judge failures"):
        validate_frozen_reference_scores(frame)
    frame.loc[1, "industrial_reference_status"] = "usable"
    frame.loc[1, "quality_score"] = 70
    validate_frozen_reference_scores(frame)


def test_old_industrial_data_keeps_existing_nullable_score_behavior():
    frame = pd.DataFrame([{"index": "old", "quality_score": None,
                           "industrial_reference_status": "conflict"}])
    validate_frozen_reference_scores(frame)
