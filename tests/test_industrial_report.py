import pandas as pd
import pytest

from eval_tool.industrial_report import summarize_industrial


def test_language_groups_and_nullable_scores_do_not_change_other_datasets():
    details = pd.DataFrame([
        {"model": "base", "dataset": "industrial_zh", "quality_score": 80,
         "location_score": None, "meta.defect_type": "crack"},
        {"model": "base", "dataset": "industrial_zh", "quality_score": 60,
         "location_score": 40, "meta.defect_type": "crack"},
        {"model": "base", "dataset": "industrial_en", "quality_score": 90,
         "location_score": None, "meta.defect_type": "crack"},
        {"model": "base", "dataset": "book_vqa", "quality_score": 0,
         "location_score": 0, "meta.defect_type": "crack"},
    ])
    original = details.copy(deep=True)
    report = summarize_industrial(details, min_n=1, bootstrap_n=20)
    assert set(report.dataset) == {"industrial_zh", "industrial_en"}
    overall = report[report.dim == "overall"].set_index(["dataset", "metric"])
    assert overall.loc[("industrial_zh", "quality_score"), "score"] == 70
    assert overall.loc[("industrial_en", "quality_score"), "score"] == 90
    assert overall.loc[("industrial_zh", "location_score"), "n"] == 1
    assert overall.loc[("industrial_zh", "location_score"), "score"] == 40
    assert ("industrial_en", "location_score") not in overall.index
    assert "defect_type" in set(report.dim)
    pd.testing.assert_frame_equal(details, original)


def test_small_attribute_groups_are_flagged():
    details = pd.DataFrame([{
        "model": "base", "dataset": "industrial_zh", "quality_score": 100,
        "meta.defect_type": "scratch",
    }])
    report = summarize_industrial(details, bootstrap_n=20)
    assert set(report.status) == {"insufficient"}
    assert report.score.isna().all()
    with pytest.raises(ValueError, match="no industrial"):
        summarize_industrial(details, datasets=("missing",))


def test_shared_combined_export_category_is_the_object_dimension(tmp_path):
    from eval_tool.report import write_reports

    source = pd.DataFrame([{
        "model": "base", "dataset": "industrial_zh", "index": "p1__t1",
        "category": "bottle", "hit": 1, "quality_score": 85,
        "meta.object_category": "bottle", "meta.defect_type": "crack",
    }])
    written = write_reports(tmp_path, [source], None, "base", 20, 42,
                            do_length_control=False, min_category_n=1)
    combined = pd.read_excel(written["judge_detail_all.xlsx"])
    assert "meta.object_category" not in combined
    result = summarize_industrial(combined, min_n=1, bootstrap_n=20)
    by_object = result[(result.dim == "object_category") & (result.metric == "quality_score")]
    assert by_object.iloc[0]["value"] == "bottle"
    assert by_object.iloc[0]["score"] == 85
    full = pd.read_excel(written["detail_base_industrial_zh.xlsx"])
    full_result = summarize_industrial(full, min_n=1, bootstrap_n=20)
    assert "defect_type" in set(full_result.dim)


def test_unscored_references_are_visible_without_becoming_zero_scores(tmp_path):
    from eval_tool.industrial_report import summarize_industrial_status

    details = pd.DataFrame([
        {"model": "base", "dataset": "industrial_en", "category": "bottle",
         "quality_score": None, "hit": None,
         "industrial_reference_status": status, "judge_reason": reason}
        for status, reason in [("conflict", "reference contradicts image"),
                               ("insufficient", "image unclear"),
                               (None, "[judge_error] timeout"),
                               (None, "[missing_prediction]")]
    ])
    scores = summarize_industrial(details, min_n=1, bootstrap_n=20)
    assert scores.empty and "score" in scores.columns
    status = summarize_industrial_status(details)
    overall = status[status.dim == "overall"].iloc[0]
    assert overall.n_total == 4 and overall.n_scored == 0
    assert overall.n_reference_conflict == 1
    assert overall.n_reference_insufficient == 1
    assert overall.n_judge_error == 1 and overall.n_missing_prediction == 1
    assert overall.n_other_unscored == 0

    from eval_tool.report import write_reports

    written = write_reports(tmp_path, [details], None, "base", 20, 42,
                            do_length_control=False,
                            dataset_kinds={"industrial_en": "industrial_describe"})
    assert pd.read_csv(written["industrial_quality_scores.csv"]).empty
    assert pd.read_csv(written["industrial_judge_status.csv"]).n_total.max() == 4
