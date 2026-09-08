"""端到端：test.jsonl 评估集 + 预测文件 -> 达标率报表。"""

import json

import pandas as pd
import pytest

from eval_tool.config import EvalConfig, ModelConfig
from eval_tool.run_eval import run


def _record(index, box, task="ground_appearance", difficulty="easy", size="large"):
    return {
        "id": f"img{index}_{task}_{index}",
        "images": [f"img{index}.jpg"],
        "conversations": [
            {"from": "human", "value": "<image>\n请给出银灰色的三轮车的边界框。"},
            {"from": "gpt", "value": json.dumps({"bbox_2d": box, "label": "三轮车"}, ensure_ascii=False)},
            {"from": "human", "value": "这辆三轮车本身长什么样？"},
            {"from": "gpt", "value": "深红色车身。"},
        ],
        "metadata": {
            "task_type": task, "difficulty": difficulty, "size_bucket": size,
            "describe_kind": "appearance", "bbox_scale": 1000, "label": "三轮车",
            "source_image": f"img{index}.jpg", "n_turns": 2,
        },
    }


@pytest.fixture
def workspace(tmp_path):
    tsv_dir = tmp_path / "data"
    tsv_dir.mkdir()
    records = [_record(i, [100, 100, 600, 600]) for i in range(4)]
    (tsv_dir / "eval_set_v1.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )
    return tmp_path, tsv_dir


def _run(tmp_path, tsv_dir, predictions, **params):
    pred_path = tmp_path / "sft_pred.csv"
    pd.DataFrame(predictions).to_csv(pred_path, index=False)
    return run(
        EvalConfig(
            tsv_dir=tsv_dir,
            out_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            datasets={"ground_box": "eval_set_v1"},
            dataset_kinds={"ground_box": "grounding_single"},
            dataset_params={"ground_box": {"select": [{"turn": 1}], **params}},
            models=[ModelConfig(name="sft", paths={"ground_box": str(pred_path)})],
            baseline_model="sft",
            enabled_datasets=["ground_box"],
            do_pointwise=False,
            do_pairwise=False,
            bootstrap_n=20,
        )
    )


def test_pass_rate_lands_in_the_summary_with_a_confidence_interval(workspace):
    tmp_path, tsv_dir = workspace
    written = _run(
        tmp_path,
        tsv_dir,
        [
            {"index": "img0_ground_appearance_0__t1", "prediction": '{"bbox_2d":[102,101,598,603]}'},
            {"index": "img1_ground_appearance_1__t1", "prediction": '{"bbox_2d":[105,105,595,595]}'},
            {"index": "img2_ground_appearance_2__t1", "prediction": '{"bbox_2d":[700,700,900,900]}'},
            {"index": "img3_ground_appearance_3__t1", "prediction": "图中没有这样的目标。"},
        ],
    )
    summary = pd.read_csv(written["report_summary.csv"])
    assert summary.loc[0, "ground_box:overall"] == 0.5      # 4 条里 2 条达标
    assert summary.loc[0, "ground_box:overall:n"] == 4      # 分母是全部样本
    assert summary.loc[0, "ground_box:overall:ci_low"] <= 0.5 <= summary.loc[0, "ground_box:overall:ci_high"]

    detail = pd.read_excel(tmp_path / "out" / "detail_sft_ground_box.xlsx")
    assert sorted(detail["fail_bucket"].fillna("")) == ["", "", "localize_fail", "malformed"]
    # 合规列一起挂上了
    assert detail["format_ok"].tolist() == [1, 1, 1, 0]
    assert detail["task_bleed"].sum() == 1


def test_only_the_selected_turn_is_scored(workspace):
    """轮 2 是描述，用另一个打分器；混进单框数据集会整批记成格式不合规。"""
    tmp_path, tsv_dir = workspace
    written = _run(
        tmp_path,
        tsv_dir,
        [{"index": f"img{i}_ground_appearance_{i}__t1", "prediction": '{"bbox_2d":[102,101,598,603]}'}
         for i in range(4)],
    )
    detail = pd.read_excel(tmp_path / "out" / "detail_sft_ground_box.xlsx")
    assert len(detail) == 4
    assert set(detail["turn"]) == {1}


def test_threshold_params_reach_the_scorer_through_the_config(workspace):
    tmp_path, tsv_dir = workspace
    predictions = [
        {"index": f"img{i}_ground_appearance_{i}__t1", "prediction": '{"bbox_2d":[120,120,620,620]}'}
        for i in range(4)
    ]
    loose = _run(tmp_path, tsv_dir, predictions, dev_threshold_pct=5.0)
    assert pd.read_csv(loose["report_summary.csv"]).loc[0, "ground_box:overall"] == 1.0
    strict = _run(tmp_path, tsv_dir, predictions, dev_threshold_pct=1.0)
    assert pd.read_csv(strict["report_summary.csv"]).loc[0, "ground_box:overall"] == 0.0
