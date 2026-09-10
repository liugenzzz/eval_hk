from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def prediction_stems(datasets: Mapping[str, str]) -> dict[str, str]:
    """dataset_key -> 预测文件名的主干。

    名字唯一时就是名字本身 —— 装备/书籍那些配置里一个键对一个评估集，文件名一个
    字节都不变（``base_aero_vqa.xlsx``）。

    几个键**共用同一个评估集**时必须缀上键名。目标检测那八个主线数据集读的都是
    ``eval_set_v1``，各自 select 出一个子集；按名字命名的话它们会写到同一个
    ``sft_eval_set_v1.xlsx`` 里 —— 第一个数据集推完，后面七个看见文件已存在就直接
    跳过，报表拿到的是**另一个数据集的预测**。这不是会报错的那种坏，是数字看着
    正常但全错的那种。
    """
    counts = Counter(datasets.values())
    return {
        key: name if counts[name] == 1 else f"{name}__{key}"
        for key, name in datasets.items()
    }


@dataclass(frozen=True)
class ArtifactLayout:
    work_dir: Path
    out_dir: Path

    def model_dir(self, model_name: str) -> Path:
        return self.work_dir / model_name

    def prediction(self, model_name: str, dataset_name: str) -> Path:
        return self.model_dir(model_name) / f"{model_name}_{dataset_name}.xlsx"

    def partial_dir(self, model_name: str) -> Path:
        return self.model_dir(model_name) / "_partial"

    def manifest(self, model_name: str, dataset_name: str) -> Path:
        return self.model_dir(model_name) / f"{model_name}_{dataset_name}.infer.json"

    def rubric_out(self, rubric: str) -> Path:
        return self.out_dir.with_name(f"{self.out_dir.name}_{rubric}")
