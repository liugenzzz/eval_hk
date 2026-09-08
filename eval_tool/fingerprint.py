"""§13 评估集冻结与四指纹硬校验。

checkpoint 数量待定（看训练效果决定评几个）。正因为不定，评估集必须冻结 —— 否则
不同时间评的 checkpoint 之间没有可比性。每份评估结果记四个指纹：

==================  ====================================================
``eval_set_sha``     评估集内容哈希（逐数据集算，抽样一次后就不该再动）
``profile_version``  打分口径版本（阈值、判据改了就该动它）
``rubric_version``   裁判提示词版本（这里取 judge 指纹：模型 + 温度 + 两份提示词）
``judge_model``      裁判模型标识
==================  ====================================================

**硬校验：四者任一不同的结果，不许画进同一张对比图。** 最容易出事的是复用
``scored`` 路径的那条通路 —— 上一轮用旧 rubric 打的 base 结果，和这一轮用新 rubric
打的 sft 结果放进同一张表，差值里混着口径变化，那不是模型的差别。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

EVAL_SET_SHA = "fp.eval_set_sha"
PROFILE_VERSION = "fp.profile_version"
RUBRIC_VERSION = "fp.rubric_version"
JUDGE_MODEL = "fp.judge_model"

FINGERPRINT_COLUMNS = (EVAL_SET_SHA, PROFILE_VERSION, RUBRIC_VERSION, JUDGE_MODEL)


class FingerprintMismatch(ValueError):
    """同一个数据集上，不同模型的结果带着不同的指纹。"""


@dataclass(frozen=True)
class RunFingerprint:
    eval_set_sha: str
    profile_version: str
    rubric_version: str
    judge_model: str

    def as_columns(self) -> dict[str, str]:
        return {
            EVAL_SET_SHA: self.eval_set_sha,
            PROFILE_VERSION: self.profile_version,
            RUBRIC_VERSION: self.rubric_version,
            JUDGE_MODEL: self.judge_model,
        }

    def to_dict(self) -> dict[str, str]:
        return {
            "eval_set_sha": self.eval_set_sha,
            "profile_version": self.profile_version,
            "rubric_version": self.rubric_version,
            "judge_model": self.judge_model,
        }


def stamp(frame: pd.DataFrame, fingerprint: RunFingerprint) -> pd.DataFrame:
    """给刚打完分的表盖指纹。已经带指纹的（复用的 scored 文件）不覆盖 ——
    覆盖掉就等于把「它是用旧口径打的」这件事抹了，校验也就查不出来了。"""
    out = frame.copy()
    for column, value in fingerprint.as_columns().items():
        if column not in out.columns:
            out[column] = value
    return out


def check_comparable(details: pd.DataFrame) -> None:
    """同一个数据集上的所有模型必须带同一组指纹，否则拒绝出对比报表。"""
    present = [col for col in FINGERPRINT_COLUMNS if col in details.columns]
    if not present or "dataset" not in details.columns or "model" not in details.columns:
        return
    for dataset, group in details.groupby("dataset", dropna=False):
        combos = group[["model", *present]].drop_duplicates()
        distinct = combos[present].drop_duplicates()
        if len(distinct) <= 1:
            continue
        lines = [
            "  " + " / ".join(f"{col.removeprefix('fp.')}={row[col]}" for col in present)
            + f"  <- {row['model']}"
            for _, row in combos.iterrows()
        ]
        raise FingerprintMismatch(
            f"数据集 {dataset} 上的结果指纹不一致，不能画进同一张对比图：\n"
            + "\n".join(lines)
            + "\n差值里会混进口径变化，那不是模型的差别。要么用同一套口径重跑，"
            "要么把它们分成两次评估。"
        )


def from_settings(
    eval_set_sha: str,
    profile_version: str,
    judge_settings: Any,
    rubric_version: str | None = None,
) -> RunFingerprint:
    return RunFingerprint(
        eval_set_sha=str(eval_set_sha or ""),
        profile_version=str(profile_version or ""),
        rubric_version=str(rubric_version or getattr(judge_settings, "fingerprint", "") or ""),
        judge_model=str(getattr(judge_settings, "model", "") or ""),
    )


def dataset_shas(shas: Mapping[str, str]) -> str:
    """多个数据集时把各自的哈希拼成一个可读的串。"""
    return ",".join(f"{key}:{value[:12]}" for key, value in sorted(shas.items()))
