"""Aircraft-specific metrics with image-level coverage gates."""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from .aggregate import bootstrap_mean_ci


CLASSES = ("fastener_missing", "crack", "corrosion", "dent", "scratch")
ACCEPTANCE_DATASETS = (
    "aircraft_presence", "aircraft_category", "aircraft_grounding",
    "aircraft_refusal", "aircraft_region",
)
INDUSTRIAL_IMAGE_DATASETS = frozenset({
    "visa", "real_iad", "mvtec_ad_screw", "mvtec_loco_screwbag",
})
AIRCRAFT_IMAGE_DATASETS = frozenset({
    "aircraft_skin_defects", "uts_aircraft_defect",
    "mm_general_aircraft_image_level", "sutd_aircraft_ai_v4",
    "priority_aircraft_curated", "corrosion_cs_vt",
    "synthetic_panel", "synthetic_closeup",
})


def summarize_aircraft_image_groups(
    details: pd.DataFrame, kinds: Mapping[str, str],
) -> pd.DataFrame:
    """Report v2 image provenance separately without changing its headline score.

    Each scored turn is one observation, matching the existing task summaries.
    ``n_records`` additionally counts original questions so multi-turn examples do
    not masquerade as independent questions.
    """
    keys = {key for key, kind in kinds.items() if kind.startswith("aircraft_")}
    if details.empty or not keys:
        return pd.DataFrame()
    data = details.loc[details["dataset"].isin(keys)].copy()
    if data.empty:
        return pd.DataFrame()
    source_col = "meta.source_dataset"
    if source_col not in data:
        raise ValueError("Aircraft subgroup report requires meta.source_dataset")
    sources = data[source_col].dropna().unique()
    unknown = sorted({str(s) for s in sources} - INDUSTRIAL_IMAGE_DATASETS - AIRCRAFT_IMAGE_DATASETS)
    if data[source_col].isna().any() or unknown:
        raise ValueError(f"Unclassified aircraft image sources: {unknown}; missing={data[source_col].isna().sum()}")
    data["image_group"] = data[source_col].map(
        lambda s: "industrial_image" if s in INDUSTRIAL_IMAGE_DATASETS else "aircraft_image"
    )
    data["record_id"] = data["index"].astype(str).str.replace(r"__t\d+$", "", regex=True)
    rows = []
    for (model, image_group, dataset), frame in data.groupby(
        ["model", "image_group", "dataset"], dropna=False
    ):
        for metric in ("hit", "quality_score"):
            if metric not in frame:
                continue
            valid = pd.to_numeric(frame[metric], errors="coerce").dropna()
            if valid.empty:
                continue
            rows.append({
                "model": model, "image_group": image_group, "dataset": dataset,
                "metric": metric, "score": round(float(valid.mean()), 4) if len(valid) else math.nan,
                "n_scored": len(valid), "n_turns": len(frame),
                "n_records": frame["record_id"].nunique(),
            })
    for (model, image_group), frame in data.groupby(["model", "image_group"], dropna=False):
        for metric in ("hit", "quality_score"):
            if metric not in frame:
                continue
            valid = pd.to_numeric(frame[metric], errors="coerce").dropna()
            if valid.empty:
                continue
            rows.append({
                "model": model, "image_group": image_group, "dataset": "all_tasks",
                "metric": metric, "score": round(float(valid.mean()), 4) if len(valid) else math.nan,
                "n_scored": len(valid), "n_turns": len(frame),
                "n_records": frame["record_id"].nunique(),
                "status": "descriptive_row_mean",
            })
        weighted = frame.loc[frame["dataset"].isin(ACCEPTANCE_DATASETS)].copy()
        scored = pd.to_numeric(weighted["hit"], errors="coerce")
        weighted = weighted.loc[scored.notna()].copy()
        weighted["hit"] = scored.loc[scored.notna()]
        per_dataset = weighted.groupby("dataset")["hit"].mean()
        missing = sorted(set(ACCEPTANCE_DATASETS) - set(per_dataset.index))
        rows.append({
            "model": model, "image_group": image_group,
            "dataset": "acceptance_score", "metric": "hit",
            "score": round(float(per_dataset.loc[list(ACCEPTANCE_DATASETS)].mean()), 4)
                     if not missing else math.nan,
            "n_scored": len(weighted), "n_turns": len(frame.loc[frame["dataset"].isin(ACCEPTANCE_DATASETS)]),
            "n_records": weighted["record_id"].nunique(),
            "n_images": weighted["source_id"].nunique() if "source_id" in weighted else math.nan,
            "status": "complete" if not missing else "missing_weighted_slice",
            "missing_datasets": ",".join(missing),
        })
    return pd.DataFrame(rows)


def _metric(
    frame: pd.DataFrame, *, model: str, name: str, column: str = "hit",
    min_images: int = 30, seed: int = 42, bootstrap_n: int = 1000,
    status_override: str | None = None,
) -> dict[str, object]:
    if frame.empty or column not in frame:
        return {"model": model, "metric": name, "score": math.nan,
                "ci_low": math.nan, "ci_high": math.nan, "n_qa": 0,
                "n_images": 0, "status": "not_in_data"}
    values = pd.to_numeric(frame[column], errors="coerce")
    table = frame.assign(_value=values).dropna(subset=["_value"])
    image_col = (
        "meta.source_image" if "meta.source_image" in table else
        "source_id" if "source_id" in table else "index"
    )
    per_image = table.groupby(image_col, dropna=False)["_value"].mean()
    n_images = len(per_image)
    status = status_override or ("ok" if n_images >= min_images else "insufficient")
    if status in {"ok", "trend_only"}:
        low, high = bootstrap_mean_ci(per_image, n_bootstrap=bootstrap_n, seed=seed)
        score = round(float(per_image.mean()), 4)
    else:
        score = low = high = math.nan
    return {"model": model, "metric": name, "score": score,
            "ci_low": low, "ci_high": high, "n_qa": len(table),
            "n_images": n_images, "status": status}


def summarize_aircraft(
    details: pd.DataFrame, kinds: Mapping[str, str], *,
    min_images: int = 30, bootstrap_n: int = 1000, seed: int = 42,
) -> pd.DataFrame:
    aircraft_keys = {key for key, kind in kinds.items() if kind.startswith("aircraft_")}
    if details.empty or not aircraft_keys:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for model, model_frame in details[details["dataset"].isin(aircraft_keys)].groupby("model"):
        slices = {kind: model_frame[model_frame["dataset"].map(kinds) == kind]
                  for kind in {kinds[key] for key in aircraft_keys}}
        presence = slices.get("aircraft_presence", pd.DataFrame())
        positive = presence[presence["gold_present"] == 1] if "gold_present" in presence else pd.DataFrame()
        negative = presence[presence["gold_present"] == 0] if "gold_present" in presence else pd.DataFrame()
        pos_images = positive["source_id"].nunique() if "source_id" in positive else 0
        neg_images = negative["source_id"].nunique() if "source_id" in negative else 0
        presence_status = ("one_sided" if not pos_images or not neg_images else
                           "insufficient" if min(pos_images, neg_images) < min_images else None)
        row = _metric(presence, model=str(model), name="defect_presence_accuracy",
                      min_images=min_images, seed=seed, bootstrap_n=bootstrap_n,
                      status_override=presence_status)
        row.update({"n_positive_images": pos_images, "n_negative_images": neg_images})
        rows.append(row)
        rows.append(_metric(negative, model=str(model), name="presence_false_positive_rate",
                            column="false_positive", min_images=min_images, seed=seed,
                            bootstrap_n=bootstrap_n))

        category = slices.get("aircraft_category", pd.DataFrame())
        per_class = []
        for name in CLASSES:
            cls = category[category["gold_class"] == name] if "gold_class" in category else pd.DataFrame()
            item = _metric(cls, model=str(model), name=f"abnormal_category_accuracy:{name}",
                           min_images=min_images, seed=seed, bootstrap_n=bootstrap_n)
            rows.append(item)
            per_class.append(item)
        valid = all(item["status"] == "ok" for item in per_class)
        rows.append(_metric(category, model=str(model), name="abnormal_category_accuracy_micro",
                            min_images=min_images, seed=seed, bootstrap_n=bootstrap_n,
                            status_override=None if valid else "trend_only"))
        rows.append({
            "model": str(model), "metric": "abnormal_category_accuracy_macro5",
            "score": round(sum(float(item["score"]) for item in per_class) / 5, 4) if valid else math.nan,
            "ci_low": math.nan, "ci_high": math.nan,
            "n_qa": sum(int(item["n_qa"]) for item in per_class),
            "n_images": sum(int(item["n_images"]) for item in per_class),
            "status": "ok" if valid else "insufficient_class_coverage",
            "undercovered_classes": ",".join(name for name, item in zip(CLASSES, per_class)
                                            if item["status"] != "ok"),
        })

        grounding = slices.get("aircraft_grounding", pd.DataFrame())
        loc = grounding[grounding["n_gt_boxes"] > 0] if "n_gt_boxes" in grounding else pd.DataFrame()
        priority_coverage = (
            all(loc.loc[loc["meta.defect_type"] == name, "source_id"].nunique() >= min_images
                for name in ("fastener_missing", "corrosion"))
            if {"meta.defect_type", "source_id"}.issubset(loc.columns) else False
        )
        grounding_status = None if priority_coverage else "trend_only"
        rows.append(_metric(loc, model=str(model), name="class_aware_localization_exact",
                            min_images=min_images, seed=seed, bootstrap_n=bootstrap_n,
                            status_override=grounding_status))
        # The exact metric above requires strict JSON as well as right boxes: a base model
        # that wraps correct boxes in ```json fences scores 0 there. The geometry-only
        # figure shows how much of the gap is localisation and how much is format.
        rows.append(_metric(loc, model=str(model), name="class_aware_localization_geometry",
                            column="geometry_hit", min_images=min_images, seed=seed,
                            bootstrap_n=bootstrap_n, status_override=grounding_status))
        rows.append(_metric(loc, model=str(model), name="class_aware_box_f1",
                            column="box_f1", min_images=min_images, seed=seed,
                            bootstrap_n=bootstrap_n, status_override=grounding_status))
        rows.append(_metric(loc, model=str(model), name="localization_format_compliance",
                            column="format_ok", min_images=min_images, seed=seed,
                            bootstrap_n=bootstrap_n, status_override=grounding_status))
        # Grounding negatives use a separate dataset key but share a scorer kind.
        refusal = model_frame[model_frame["dataset"].isin(
            key for key in aircraft_keys if key.endswith("refusal")
        )]
        rows.append(_metric(refusal, model=str(model), name="grounding_empty_answer_accuracy",
                            min_images=min_images, seed=seed, bootstrap_n=bootstrap_n))
        # The same split for negatives: ```json [] ``` is a correct "nothing here" that the
        # strict metric scores 0 (Qwen2-VL-7B on aircraft_eval_v2: 0.29 strict, 0.97 here).
        rows.append(_metric(refusal, model=str(model), name="grounding_empty_answer_geometry",
                            column="geometry_hit", min_images=min_images, seed=seed,
                            bootstrap_n=bootstrap_n))
        region = slices.get("aircraft_region", pd.DataFrame())
        rows.append(_metric(region, model=str(model), name="semantic_region_accuracy",
                            min_images=min_images, seed=seed, bootstrap_n=bootstrap_n))
    return pd.DataFrame(rows)
