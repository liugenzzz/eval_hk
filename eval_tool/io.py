from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PREDICTION_COLUMNS = (
    "prediction",
    "pred",
    "response",
    "model_answer",
    "answer_pred",
    "模型回答",
    "预测",
)


@dataclass(frozen=True)
class AlignResult:
    data: pd.DataFrame
    missing_count: int
    extra_predictions: pd.DataFrame


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype={"index": str})
    if suffix == ".csv":
        return pd.read_csv(path, dtype={"index": str})
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype={"index": str})
    raise ValueError(f"Unsupported table format: {path}")


def normalize_index(data: pd.DataFrame) -> pd.DataFrame:
    if "index" not in data.columns:
        raise ValueError("Table must contain an index column")
    out = data.copy()
    out["index"] = out["index"].map(lambda x: "" if pd.isna(x) else str(x).strip())
    return out


def detect_prediction_column(data: pd.DataFrame) -> str:
    lower = {str(col).lower(): col for col in data.columns}
    for name in PREDICTION_COLUMNS:
        if name.lower() in lower:
            return str(lower[name.lower()])
    raise ValueError(f"Prediction file lacks a prediction column. Tried: {', '.join(PREDICTION_COLUMNS)}")


def load_truth_dataset(tsv_dir: str | Path, dataset_name: str) -> pd.DataFrame:
    return normalize_index(read_table(Path(tsv_dir) / f"{dataset_name}.tsv"))


def load_prediction_file(path: str | Path) -> pd.DataFrame:
    data = normalize_index(read_table(path))
    pred_col = detect_prediction_column(data)
    if pred_col != "prediction":
        data = data.rename(columns={pred_col: "prediction"})
    return data


def align_truth_and_prediction(truth: pd.DataFrame, prediction: pd.DataFrame) -> AlignResult:
    truth = normalize_index(truth)
    prediction = normalize_index(prediction)
    pred_cols = ["index", "prediction"]
    pred = prediction[pred_cols].drop_duplicates(subset=["index"], keep="first")
    merged = truth.merge(pred, on="index", how="left")
    missing_count = int(merged["prediction"].isna().sum())
    extra = prediction[~prediction["index"].isin(set(truth["index"]))].copy()
    return AlignResult(data=merged, missing_count=missing_count, extra_predictions=extra)


def image_map_from_truth(truth: pd.DataFrame) -> dict[str, str]:
    if "image" not in truth.columns:
        return {}
    data = normalize_index(truth)
    return {
        str(row["index"]): str(row["image"])
        for _, row in data.iterrows()
        if not pd.isna(row.get("image")) and str(row.get("image", "")).strip()
    }
