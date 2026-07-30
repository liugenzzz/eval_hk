import pandas as pd
import pytest

from eval_tool.config import InferConfig
from eval_tool.run_infer import run


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, image_b64=None):
        self.calls.append((prompt, image_b64))
        return "答案是 A" if "选项" in prompt else "开放回答"


class FakeBatchGenerator:
    def __init__(self):
        self.batch_calls = []

    def generate_batch(self, prompts, image_b64s=None):
        image_b64s = image_b64s or [None] * len(prompts)
        self.batch_calls.append((list(prompts), list(image_b64s)))
        return [f"pred-{i}" for i, _ in enumerate(prompts)]


def test_run_infer_generates_reusable_xlsx_without_image_column(tmp_path):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    out_dir = tmp_path / "pred"
    prompt_path = tmp_path / "infer_mcq.txt"
    prompt_path.write_text("题目：{question}\n选项A：{A}\n选项B：{B}", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "index": "1",
                "image": "base64-image",
                "question": "选哪个？",
                "A": "燃油泵",
                "B": "滑油泵",
                "answer": "A",
                "category": "P1",
                "l2-category": "零件图",
                "source_id": "s1",
            }
        ]
    ).to_csv(tsv_dir / "aero_mcq.tsv", sep="\t", index=False)

    generator = FakeGenerator()
    written = run(
        InferConfig(
            model_name="base",
            model_path=tmp_path / "model",
            tsv_dir=tsv_dir,
            out_dir=out_dir,
            datasets={"mcq": "aero_mcq"},
            prompt_files={"mcq": prompt_path},
        ),
        generator=generator,
    )

    path = written["mcq"]
    assert path == out_dir / "base_aero_mcq.xlsx"
    output = pd.read_excel(path, dtype={"index": str})
    assert "image" not in output.columns
    assert output.loc[0, "prediction"] == "答案是 A"
    assert generator.calls == [("题目：选哪个？\n选项A：燃油泵\n选项B：滑油泵", "base64-image")]


def test_run_infer_uses_batch_generation_when_available(tmp_path):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    out_dir = tmp_path / "pred"
    prompt_path = tmp_path / "infer_vqa.txt"
    prompt_path.write_text("题目：{question}", encoding="utf-8")
    pd.DataFrame(
        [
            {"index": "1", "image": "img1", "question": "q1", "answer": "a1"},
            {"index": "2", "image": "img2", "question": "q2", "answer": "a2"},
            {"index": "3", "image": "img3", "question": "q3", "answer": "a3"},
        ]
    ).to_csv(tsv_dir / "aero_vqa.tsv", sep="\t", index=False)

    generator = FakeBatchGenerator()
    run(
        InferConfig(
            model_name="base",
            model_path=tmp_path / "model",
            tsv_dir=tsv_dir,
            out_dir=out_dir,
            datasets={"vqa": "aero_vqa"},
            prompt_files={"vqa": prompt_path},
            batch_size=2,
        ),
        generator=generator,
    )

    assert generator.batch_calls == [
        (["题目：q1", "题目：q2"], ["img1", "img2"]),
        (["题目：q3"], ["img3"]),
    ]
    output = pd.read_excel(out_dir / "base_aero_vqa.xlsx", dtype={"index": str})
    assert output["prediction"].tolist() == ["pred-0", "pred-1", "pred-0"]


def test_run_infer_uses_parallel_branch_without_creating_main_generator(tmp_path, monkeypatch):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    out_dir = tmp_path / "pred"
    pd.DataFrame([{"index": "1", "image": "img1", "question": "q1"}]).to_csv(
        tsv_dir / "aero_vqa.tsv", sep="\t", index=False
    )

    def fake_parallel(config, dataset_key, rows):
        return ["parallel-pred"]

    monkeypatch.setattr("eval_tool.run_infer._run_dataset_parallel", fake_parallel)

    run(
        InferConfig(
            model_name="base",
            model_path=tmp_path / "model",
            tsv_dir=tsv_dir,
            out_dir=out_dir,
            datasets={"vqa": "aero_vqa"},
            prompt_files={},
            gpu_ids=[0],
        )
    )

    output = pd.read_excel(out_dir / "base_aero_vqa.xlsx", dtype={"index": str})
    assert output.loc[0, "prediction"] == "parallel-pred"
