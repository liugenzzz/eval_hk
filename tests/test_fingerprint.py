"""§13 评估集冻结与四指纹硬校验 + §18.4 像素预算硬校验。"""

import json

import pandas as pd
import pytest

from eval_tool.config import ConfigError, EvalConfig, ModelConfig, load_pipeline_config
from eval_tool.fingerprint import FingerprintMismatch, RunFingerprint, check_comparable, stamp
from eval_tool.run_eval import run

FP = RunFingerprint("a" * 64, "grounding_zh_v1", "rubric-abc", "qwen3.6-27b")


def test_stamp_adds_the_four_fingerprints():
    out = stamp(pd.DataFrame([{"model": "sft"}]), FP)
    assert out.loc[0, "fp.eval_set_sha"] == "a" * 64
    assert out.loc[0, "fp.profile_version"] == "grounding_zh_v1"
    assert out.loc[0, "fp.rubric_version"] == "rubric-abc"
    assert out.loc[0, "fp.judge_model"] == "qwen3.6-27b"


def test_stamp_does_not_overwrite_an_existing_fingerprint():
    """复用的 scored 文件带着它自己的指纹。覆盖掉就等于把「它是用旧口径打的」抹了，
    校验也就查不出来了。"""
    old = pd.DataFrame([{"model": "base", "fp.rubric_version": "rubric-old"}])
    assert stamp(old, FP).loc[0, "fp.rubric_version"] == "rubric-old"


def test_mismatched_rubrics_on_one_dataset_are_refused():
    """上一轮用旧 rubric 打的 base，和这一轮用新 rubric 打的 sft 放进同一张表，
    差值里混着口径变化，那不是模型的差别。"""
    other = RunFingerprint("a" * 64, "grounding_zh_v1", "rubric-NEW", "qwen3.6-27b")
    details = pd.concat(
        [
            stamp(pd.DataFrame([{"model": "base", "dataset": "g"}]), FP),
            stamp(pd.DataFrame([{"model": "sft", "dataset": "g"}]), other),
        ],
        ignore_index=True,
    )
    with pytest.raises(FingerprintMismatch, match="指纹不一致"):
        check_comparable(details)


def test_the_same_fingerprint_across_models_is_fine():
    details = pd.concat(
        [
            stamp(pd.DataFrame([{"model": "base", "dataset": "g"}]), FP),
            stamp(pd.DataFrame([{"model": "sft", "dataset": "g"}]), FP),
        ],
        ignore_index=True,
    )
    check_comparable(details)


def test_two_datasets_may_have_different_eval_set_hashes():
    """不同数据集本来就是不同的文件，只有同一个数据集内部要求一致。"""
    other = RunFingerprint("b" * 64, "grounding_zh_v1", "rubric-abc", "qwen3.6-27b")
    details = pd.concat(
        [
            stamp(pd.DataFrame([{"model": "sft", "dataset": "g1"}]), FP),
            stamp(pd.DataFrame([{"model": "sft", "dataset": "g2"}]), other),
        ],
        ignore_index=True,
    )
    check_comparable(details)


def test_run_writes_the_fingerprint_file(tmp_path):
    tsv_dir = tmp_path / "data"
    tsv_dir.mkdir()
    pd.DataFrame(
        [{"index": "1", "image": "", "question": "q", "A": "a", "B": "b", "C": "c", "D": "d",
          "answer": "B", "category": "P1", "l2-category": "", "source_id": "s1"}]
    ).to_csv(tsv_dir / "aero_mcq.tsv", sep="\t", index=False)
    pred = tmp_path / "pred.csv"
    pd.DataFrame([{"index": "1", "prediction": "答案是 B"}]).to_csv(pred, index=False)

    written = run(
        EvalConfig(
            tsv_dir=tsv_dir, out_dir=tmp_path / "out", cache_dir=tmp_path / "cache",
            datasets={"mcq": "aero_mcq"},
            models=[ModelConfig(name="base", paths={"mcq": str(pred)})],
            baseline_model="base", enabled_datasets=["mcq"],
            do_pointwise=False, do_pairwise=False, bootstrap_n=20,
            profile_name="grounding_zh_v1", profile_version="v1",
        )
    )
    payload = json.loads(written["run_fingerprint.json"].read_text(encoding="utf-8"))
    assert payload["profile_version"] == "v1"
    assert payload["profile_name"] == "grounding_zh_v1"
    assert payload["eval_set_sha"].startswith("mcq:")
    assert payload["datasets"] == {"mcq": "aero_mcq"}


# ------------------------------------------------------------ §18.4 像素预算

def _pipeline(tmp_path, infer_extra):
    config = {
        "tsv_dir": ".",
        "datasets": {"mcq": "aero_mcq"},
        "enabled_datasets": ["mcq"],
        "models": [{"name": "base", "model_path": "models/base"}],
        "baseline_model": "base",
        "infer": {"prompt_files": {}, **infer_extra},
    }
    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_inference_pixels_matching_training_pass(tmp_path):
    path = _pipeline(tmp_path, {
        "image_min_pixels": 65536, "image_max_pixels": 589824,
        "training": {"image_min_pixels": 65536, "image_max_pixels": 589824},
    })
    config = load_pipeline_config(path)
    assert config.infer.image_max_pixels == 589824


def test_inference_pixels_differing_from_training_are_refused(tmp_path):
    """分辨率变了模型的空间精度就变了，测出来的数字不可比 —— 而这在报表上完全
    看不出来，只会表现为「这个 checkpoint 好像差一点」。"""
    path = _pipeline(tmp_path, {
        "image_min_pixels": 65536, "image_max_pixels": 1048576,
        "training": {"image_min_pixels": 65536, "image_max_pixels": 589824},
    })
    with pytest.raises(ConfigError, match="不一致"):
        load_pipeline_config(path)


def test_omitting_the_pixel_budget_while_declaring_training_values_is_refused(tmp_path):
    """省略等于按旧行为不做预缩放，那和训练时也是不一致的。"""
    path = _pipeline(tmp_path, {"training": {"image_min_pixels": 65536, "image_max_pixels": 589824}})
    with pytest.raises(ConfigError, match="不一致|没设|image_min_pixels"):
        load_pipeline_config(path)


def test_half_declared_training_block_is_refused(tmp_path):
    path = _pipeline(tmp_path, {
        "image_min_pixels": 65536, "image_max_pixels": 589824,
        "training": {"image_min_pixels": 65536},
    })
    with pytest.raises(ConfigError, match="同时给"):
        load_pipeline_config(path)


def test_configs_without_a_training_block_keep_working(tmp_path):
    path = _pipeline(tmp_path, {"image_min_pixels": 65536, "image_max_pixels": 589824})
    assert load_pipeline_config(path).infer.image_min_pixels == 65536
