import pandas as pd
import pytest

from eval_tool.config import EvalConfig, ModelConfig
from eval_tool.judge import JudgeClient
from eval_tool.run_eval import run


def test_run_eval_writes_reports_for_choice_datasets_without_vqa_api(tmp_path):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    mcq_truth = pd.DataFrame(
        [
            {
                "index": "1",
                "image": "img",
                "question": "选哪个？",
                "A": "燃油泵",
                "B": "滑油泵",
                "C": "作动筒",
                "D": "传感器",
                "answer": "B",
                "category": "P3",
                "l2-category": "零件图",
                "source_id": "s1",
            }
        ]
    )
    judge_truth = pd.DataFrame(
        [
            {
                "index": "2",
                "image": "img",
                "question": "说法正确吗？",
                "A": "正确",
                "B": "错误",
                "answer": "A",
                "category": "R1",
                "l2-category": "结构图",
                "source_id": "s2",
            }
        ]
    )
    mcq_truth.to_csv(tsv_dir / "aero_mcq.tsv", sep="\t", index=False)
    judge_truth.to_csv(tsv_dir / "aero_judge.tsv", sep="\t", index=False)

    mcq_pred = tmp_path / "base_mcq.csv"
    judge_pred = tmp_path / "base_judge.csv"
    pd.DataFrame([{"index": "1", "prediction": "答案是 B"}]).to_csv(mcq_pred, index=False)
    pd.DataFrame([{"index": "2", "prediction": "正确"}]).to_csv(judge_pred, index=False)

    written = run(
        EvalConfig(
            tsv_dir=tsv_dir,
            out_dir=out_dir,
            cache_dir=cache_dir,
            datasets={"mcq": "aero_mcq", "judge": "aero_judge"},
            models=[ModelConfig(name="base", paths={"mcq": str(mcq_pred), "judge": str(judge_pred)})],
            baseline_model="base",
            do_pointwise=False,
            do_pairwise=False,
            bootstrap_n=20,
        )
    )

    assert "report_summary.csv" in written
    assert (out_dir / "detail_base_mcq.xlsx").exists()
    assert (out_dir / "detail_base_judge.xlsx").exists()
    summary = pd.read_csv(out_dir / "report_summary.csv")
    assert summary.loc[0, "mcq:overall"] == 1.0
    assert summary.loc[0, "judge:overall"] == 1.0


def test_run_eval_reuses_previously_scored_vqa_results_without_calling_judge(tmp_path, monkeypatch):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    rows = [
        {
            "index": str(index),
            "image": "img",
            "question": f"描述这张图 {index}",
            "answer": "参考答案",
            "category": "单轮",
            "l2-category": "",
            "source_id": f"s{index}",
        }
        for index in range(30)
    ]
    vqa_truth = pd.DataFrame(rows)
    vqa_truth.to_csv(tsv_dir / "aero_vqa.tsv", sep="\t", index=False)

    # Simulate a detail_base_vqa.xlsx produced by a previous run: already has hit/judge_reason.
    scored_path = tmp_path / "detail_base_vqa.xlsx"
    pd.DataFrame(
        [
            {
                "index": row["index"],
                "question": row["question"],
                "answer": row["answer"],
                "prediction": "旧结果",
                "category": "单轮",
                "l2-category": "",
                "hit": 1,
                "judge_reason": "reused",
                "pred_len": 3,
            }
            for row in rows
        ]
    ).to_excel(scored_path, index=False)

    def _boom(self, *args, **kwargs):
        raise AssertionError("judge API must not be called when reusing a scored path")

    monkeypatch.setattr(JudgeClient, "judge_pointwise", _boom)

    written = run(
        EvalConfig(
            tsv_dir=tsv_dir,
            out_dir=out_dir,
            cache_dir=cache_dir,
            datasets={"vqa": "aero_vqa"},
            models=[ModelConfig(name="base", scored_paths={"vqa": str(scored_path)})],
            baseline_model="base",
            enabled_datasets=["vqa"],
            do_pointwise=True,
            do_pairwise=False,
            bootstrap_n=20,
        )
    )

    assert "score_summary.csv" in written
    summary = pd.read_csv(written["score_summary.csv"])
    assert summary.loc[0, "total_score"] == 1.0


def test_run_eval_pairwise_works_when_baseline_uses_a_reused_scored_path(tmp_path, monkeypatch):
    """Regression test: reusing a baseline's scored xlsx (numeric-looking "index" values)
    must not break the pairwise merge against a freshly-scored challenger model, which
    happened because plain pd.read_excel infers "1" as int64 while every other dataframe
    in the pipeline keeps "index" as str."""
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / "cache"

    vqa_truth = pd.DataFrame(
        [{"index": "1", "image": "img", "question": "描述这张图", "answer": "参考答案", "category": "单轮", "l2-category": "", "source_id": "s1"}]
    )
    vqa_truth.to_csv(tsv_dir / "aero_vqa.tsv", sep="\t", index=False)

    scored_path = tmp_path / "detail_base_vqa.xlsx"
    pd.DataFrame(
        [
            {
                "index": "1",
                "question": "描述这张图",
                "answer": "参考答案",
                "prediction": "base 结果",
                "category": "单轮",
                "l2-category": "",
                "hit": 1,
                "judge_reason": "reused",
                "pred_len": 3,
            }
        ]
    ).to_excel(scored_path, index=False)

    sft_pred_path = tmp_path / "sft_vqa.xlsx"
    pd.DataFrame([{"index": "1", "prediction": "sft 结果"}]).to_excel(sft_pred_path, index=False)

    def fake_pointwise(self, question, reference, prediction, image_b64=None):
        return {
            "hit": 1,
            "reason": "ok",
            "quality_score": 90.0,
            "param_score": 90,
            "fact_score": 90,
            "visual_score": 90,
            "fabrication_score": 90,
            "style_score": 90,
        }

    def fake_pairwise(self, question, reference, answer_a, answer_b, image_b64=None):
        return "A", "ok"

    monkeypatch.setattr(JudgeClient, "judge_pointwise", fake_pointwise)
    monkeypatch.setattr(JudgeClient, "judge_pairwise", fake_pairwise)

    written = run(
        EvalConfig(
            tsv_dir=tsv_dir,
            out_dir=out_dir,
            cache_dir=cache_dir,
            datasets={"vqa": "aero_vqa"},
            models=[
                ModelConfig(name="base", scored_paths={"vqa": str(scored_path)}),
                ModelConfig(name="sft", paths={"vqa": str(sft_pred_path)}),
            ],
            baseline_model="base",
            enabled_datasets=["vqa"],
            do_pointwise=True,
            do_pairwise=True,
            bootstrap_n=20,
        )
    )

    assert "pairwise_vs_baseline.csv" in written
    pairwise = pd.read_csv(written["pairwise_vs_baseline.csv"])
    assert (pairwise["model"] == "sft").all()
    assert (pairwise["baseline_model"] == "base").all()
