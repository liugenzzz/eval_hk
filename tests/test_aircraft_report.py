import math

import pandas as pd

from eval_tool.aircraft_report import summarize_aircraft, summarize_aircraft_image_groups


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


def test_image_groups_count_original_records_and_preserve_task_scores():
    details = pd.DataFrame([
        {"model": "m", "dataset": "aircraft_presence", "index": "a__t1",
         "meta.source_dataset": "uts_aircraft_defect", "hit": 1},
        {"model": "m", "dataset": "aircraft_dialog_check", "index": "b__t1",
         "meta.source_dataset": "real_iad", "hit": 0},
        {"model": "m", "dataset": "aircraft_dialog_ground", "index": "b__t2",
         "meta.source_dataset": "real_iad", "hit": 1},
    ])
    groups = summarize_aircraft_image_groups(details, {
        "aircraft_presence": "aircraft_presence",
        "aircraft_dialog_check": "aircraft_presence",
        "aircraft_dialog_ground": "aircraft_grounding",
    })
    industrial = groups[(groups.image_group == "industrial_image") &
                        (groups.dataset == "all_tasks") & (groups.metric == "hit")].iloc[0]
    assert industrial.score == 0.5
    assert industrial.n_turns == 2
    assert industrial.n_records == 1
    aircraft = groups[(groups.image_group == "aircraft_image") &
                      (groups.dataset == "all_tasks") & (groups.metric == "hit")].iloc[0]
    assert aircraft.score == 1.0


def test_image_groups_reject_unknown_source():
    import pytest

    details = pd.DataFrame([{"model": "m", "dataset": "aircraft_presence",
                             "index": "a__t1", "meta.source_dataset": "unknown", "hit": 1}])
    with pytest.raises(ValueError, match="Unclassified aircraft image sources"):
        summarize_aircraft_image_groups(details, KINDS)


def test_image_group_acceptance_equal_weights_tasks_despite_different_sizes():
    rows = []
    scores = {"aircraft_presence": (1, 20), "aircraft_category": (0, 1),
              "aircraft_grounding": (1, 1), "aircraft_refusal": (0, 1),
              "aircraft_region": (1, 1)}
    for dataset, (hit, n) in scores.items():
        rows.extend({"model": "m", "dataset": dataset, "index": f"{dataset}-{i}__t1",
                     "source_id": f"{dataset}-{i}", "meta.source_dataset": "real_iad",
                     "hit": hit} for i in range(n))
    rows.append({"model": "m", "dataset": "aircraft_pair", "index": "pair__t1",
                 "source_id": "pair", "meta.source_dataset": "real_iad", "hit": 0})
    kinds = {name: name for name in scores}
    kinds["aircraft_pair"] = "aircraft_presence"
    report = summarize_aircraft_image_groups(pd.DataFrame(rows), kinds)
    acceptance = report[(report.image_group == "industrial_image") &
                        (report.dataset == "acceptance_score")].iloc[0]
    descriptive = report[(report.image_group == "industrial_image") &
                         (report.dataset == "all_tasks") & (report.metric == "hit")].iloc[0]
    assert acceptance.score == 0.6
    assert acceptance.n_scored == 24
    assert acceptance.n_records == 24
    assert acceptance.n_images == 24
    assert acceptance.status == "complete"
    assert descriptive.score == 22 / 25


def test_image_group_acceptance_requires_every_weighted_slice():
    details = pd.DataFrame([{"model": "m", "dataset": "aircraft_presence",
                             "index": "a__t1", "source_id": "a",
                             "meta.source_dataset": "visa", "hit": 1}])
    report = summarize_aircraft_image_groups(details, {"aircraft_presence": "aircraft_presence"})
    acceptance = report[report.dataset == "acceptance_score"].iloc[0]
    assert math.isnan(acceptance.score)
    assert acceptance.status == "missing_weighted_slice"
    assert "aircraft_category" in acceptance.missing_datasets
