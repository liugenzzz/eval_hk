"""Offline smoke check for the unified convert -> infer -> eval pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).parent


def step(message: str) -> None:
    print(f"\n===== {message} =====", flush=True)


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    try:
        from PIL import Image

        Image.new("RGB", (32, 32), color).save(path, format="JPEG")
    except ImportError:
        path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9")


def build_sample_json(data_dir: Path) -> Path:
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    first = image_dir / "1.jpg"
    second = image_dir / "2.jpg"
    third = image_dir / "3.jpg"
    make_image(first, (200, 30, 30))
    make_image(second, (30, 200, 30))
    make_image(third, (30, 30, 200))
    records = [
        {
            "id": "vqa_000001",
            "image": [str(first)],
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe the vessel."},
                {"from": "gpt", "value": "It is a survey vessel."},
            ],
        },
        {
            "id": "vqa_000002",
            "image": [str(second), str(third)],
            "conversations": [
                {"from": "human", "value": "<image>\nIdentify the aircraft."},
                {"from": "gpt", "value": "It is a transport helicopter."},
                {"from": "human", "value": "<image>\nDescribe the rotor layout."},
                {"from": "gpt", "value": "It has tandem rotors."},
            ],
        },
        {
            "id": "vqa_000003",
            "image": [str(first), str(second)],
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\n<image>\nCompare the equipment types.",
                },
                {
                    "from": "gpt",
                    "value": "The first is a vessel and the second is a helicopter.",
                },
            ],
        },
    ]
    source = data_dir / "sample_sharegpt.json"
    source.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return source


class FakeGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.seen: list[object] = []

    def generate_batch(self, prompts, image_b64s=None):
        predictions = []
        for prompt in prompts:
            self.seen.append(prompt)
            turn_type = "multi" if isinstance(prompt, list) else "single"
            predictions.append(f"[{self.model_name}] {turn_type}-turn answer")
        return predictions


def fake_judge_post(
    self,
    system_prompt: str,
    user_text: str,
    image_b64=None,
    mime: str = "image/jpeg",
) -> str:
    if "winner" in system_prompt.lower() or "回答A" in user_text:
        return '{"winner": "A", "reason": "e2e stub"}'
    return '{"correct": true, "reason": "e2e stub"}'


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="eval-tool-e2e-") as temporary:
        work = Path(temporary)
        step("1/3 build ShareGPT fixture and unified config")
        source = build_sample_json(work / "data")
        pipeline_path = work / "pipeline.json"
        pipeline_path.write_text(
            json.dumps(
                {
                    "convert": {"input_json": str(source)},
                    "tsv_dir": str(work / "LMUData"),
                    "datasets": {"vqa": "aero_vqa"},
                    "enabled_datasets": ["vqa"],
                    "work_dir": str(work / "work_dir"),
                    "out_dir": str(work / "eval_report"),
                    "cache_dir": str(work / "eval_cache"),
                    "models": [
                        {"name": "base", "model_path": "unused/base"},
                        {"name": "sft", "model_path": "unused/sft"},
                    ],
                    "baseline_model": "base",
                    "infer": {
                        "prompt_files": {
                            "vqa": str(ROOT / "prompts" / "infer_vqa_raw.txt")
                        },
                        "batch_size": 2,
                    },
                    "max_workers": 4,
                    "bootstrap_n": 50,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        step("2/3 run unified convert -> infer -> eval with fakes")
        from eval_tool import judge as judge_module
        from eval_tool.config import load_pipeline_config
        from eval_tool.pipeline import run_all

        judge_module.JudgeClient._post = fake_judge_post
        generators: dict[str, FakeGenerator] = {}

        def generator_factory(model_name: str) -> FakeGenerator:
            generator = FakeGenerator(model_name)
            generators[model_name] = generator
            return generator

        result = run_all(
            load_pipeline_config(pipeline_path),
            generator_factory=generator_factory,
        )

        import pandas as pd

        tsv_path = work / "LMUData" / "aero_vqa.tsv"
        tsv_text = tsv_path.read_text(encoding="utf-8-sig")
        assert len(tsv_text.strip().splitlines()) - 1 == 4
        assert list(generators) == ["base", "sft"]
        for model_name, generator in generators.items():
            assert len(generator.seen) == 4
            multi_turn = [prompt for prompt in generator.seen if isinstance(prompt, list)]
            assert len(multi_turn) == 1
            assert [turn["role"] for turn in multi_turn[0]] == [
                "user",
                "assistant",
                "user",
            ]
            prediction_path = result["infer"][model_name]["vqa"]
            prediction = pd.read_excel(prediction_path)
            assert "image" not in prediction.columns
            assert prediction["prediction"].notna().all()

        step("3/3 verify reports")
        written = result["eval"]
        score = pd.read_csv(written["score_summary.csv"])
        assert "total_score" in score.columns
        assert len(score) == 2
        combined = pd.read_excel(written["judge_detail_all.xlsx"])
        assert set(combined["model"]) == {"base", "sft"}
        assert "image" not in combined.columns
        assert "judge_reason" in combined.columns
        pairwise = pd.read_csv(written["pairwise_vs_baseline.csv"])
        assert not pairwise.empty

        print("\nOK: unified convert -> infer -> eval offline path")
        return 0


if __name__ == "__main__":
    sys.exit(main())
