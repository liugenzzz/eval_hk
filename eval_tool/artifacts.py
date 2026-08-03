from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
