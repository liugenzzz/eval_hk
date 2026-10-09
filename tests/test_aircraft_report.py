import math

import pandas as pd

from eval_tool.aircraft_report import summarize_aircraft


KINDS = {
    "aircraft_presence": "aircraft_presence",
    "aircraft_category": "aircraft_category",
    "aircraft_grounding": "aircraft_grounding",
    "aircraft_refusal": "aircraft_grounding",
    "aircraft_region": "aircraft_region",
}


def row_for(metrics, name):
    return metrics[metrics["metric"] == name].iloc[0]


def test_presence_with_no_normal_aircraft_images_is_not_an_acceptance_score():
    rows = [
        {"model": "m", "dataset": "aircraft_presence", "source_id": f"img{i}",
         "gold_present": 1, "hit": 1, "false_positive": 0}
        for i in range(40)
    ]
    report = summarize_aircraft(pd.DataFrame(rows), KINDS)
    presence = row_for(report, "defect_presence_accuracy")
    assert presence["status"] == "one_sided"
    assert math.isnan(presence["score"])
    assert presence["n_positive_images"] == 40
    assert presence["n_negative_images"] == 0


def test_macro_category_requires_coverage_for_every_class():
    names = ["fastener_missing", "crack", "corrosion", "dent", "scratch"]
    rows = [
        {"model": "m", "dataset": "aircraft_category", "source_id": f"{name}-{i}",
         "gold_class": name, "hit": 1}
        for name in names for i in range(30 if name != "scratch" else 29)
    ]
    report = summarize_aircraft(pd.DataFrame(rows), KINDS)
    macro = row_for(report, "abnormal_category_accuracy_macro5")
    assert macro["status"] == "insufficient_class_coverage"
    assert macro["undercovered_classes"] == "scratch"
    assert math.isnan(macro["score"])
    assert row_for(report, "abnormal_category_accuracy_micro")["score"] == 1.0


def test_coverage_counts_images_not_repeat_questions():
    rows = [
        {"model": "m", "dataset": "aircraft_region", "source_id": "same-image",
         "hit": 1}
        for _ in range(40)
    ]
    report = summarize_aircraft(pd.DataFrame(rows), KINDS)
    region = row_for(report, "semantic_region_accuracy")
    assert region["n_qa"] == 40
    assert region["n_images"] == 1
    assert region["status"] == "insufficient"


def test_format_and_geometry_are_reported_apart():
    # ```json fences fail the strict metrics; the geometry-only figures still count
    # right boxes and a right "nothing here"
    rows = [
        {"model": "m", "dataset": "aircraft_grounding", "source_id": f"pos{i}",
         "n_gt_boxes": 1, "hit": 0, "geometry_hit": 1, "format_ok": 0, "box_f1": 1.0}
        for i in range(40)
    ] + [
        {"model": "m", "dataset": "aircraft_refusal", "source_id": f"neg{i}",
         "n_gt_boxes": 0, "hit": 0, "geometry_hit": 1}
        for i in range(40)
    ]
    report = summarize_aircraft(pd.DataFrame(rows), KINDS)
    assert row_for(report, "class_aware_localization_exact")["score"] == 0.0
    assert row_for(report, "class_aware_localization_geometry")["score"] == 1.0
    assert row_for(report, "grounding_empty_answer_accuracy")["score"] == 0.0
    assert row_for(report, "grounding_empty_answer_geometry")["score"] == 1.0


def test_non_aircraft_report_remains_untouched():
    rows = pd.DataFrame([{"model": "m", "dataset": "ground_box", "hit": 1}])
    assert summarize_aircraft(rows, {"ground_box": "grounding_single"}).empty
