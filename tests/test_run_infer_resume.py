import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from eval_tool.config import InferConfig
from eval_tool.infer_cache import write_manifest as write_manifest_atomic
from eval_tool.run_infer import run


class FailingSecondBatchGenerator:
    def __init__(self):
        self.calls = 0

    def generate_batch(self, prompts, image_b64s=None):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("simulated interruption")
        return [f"prediction:{prompt}" for prompt in prompts]


class RecordingBatchGenerator:
    def __init__(self):
        self.prompts = []

    def generate_batch(self, prompts, image_b64s=None):
        self.prompts.extend(prompts)
        return [f"prediction:{prompt}" for prompt in prompts]


def _write_vqa_dataset(tmp_path, count=3):
    tsv_dir = tmp_path / "tsv"
    tsv_dir.mkdir()
    pd.DataFrame(
        [
            {
                "index": str(index),
                "image": "",
                "question": f"q{index}",
                "answer": f"a{index}",
            }
            for index in range(count)
        ]
    ).to_csv(tsv_dir / "aero_vqa.tsv", sep="\t", index=False)
    return tsv_dir


def _resume_config(tmp_path, **overrides):
    values = {
        "model_name": "base",
        "model_path": tmp_path / "model",
        "tsv_dir": _write_vqa_dataset(tmp_path),
        "out_dir": tmp_path / "pred",
        "datasets": {"vqa": "aero_vqa"},
        "prompt_files": {},
        "batch_size": 1,
        "overwrite": False,
        "resume": True,
    }
    values.update(overrides)
    return InferConfig(**values)


def test_resume_mode_does_not_implicitly_authorize_overwrite(tmp_path):
    config = InferConfig(
        model_name="base",
        model_path=tmp_path / "model",
        tsv_dir=_write_vqa_dataset(tmp_path),
        out_dir=tmp_path / "pred",
        datasets={"vqa": "aero_vqa"},
        prompt_files={},
        resume=True,
    )

    assert config.overwrite is None


def test_resume_only_generates_missing_rows_after_interruption(tmp_path):
    config = _resume_config(tmp_path)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(config, generator=FailingSecondBatchGenerator())

    partials = list((config.out_dir / "_partial").glob("vqa__*.jsonl"))
    assert len(partials) == 1
    first_record = json.loads(partials[0].read_text(encoding="utf-8").strip())
    assert first_record["index"] == "0"

    resumed = RecordingBatchGenerator()
    written = run(config, generator=resumed)

    assert len(resumed.prompts) == 2
    output = pd.read_excel(written["vqa"], dtype={"index": str})
    assert output["index"].tolist() == ["0", "1", "2"]
    assert output["prediction"].notna().all()
    assert written["vqa"].with_suffix(".infer.json").is_file()


def test_prompt_fingerprint_change_requires_explicit_overwrite(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("first {question}", encoding="utf-8")
    config = _resume_config(tmp_path, prompt_files={"vqa": prompt_path})
    run(config, generator=RecordingBatchGenerator())

    prompt_path.write_text("changed {question}", encoding="utf-8")
    blocked_generator = RecordingBatchGenerator()
    with pytest.raises(RuntimeError, match="--overwrite"):
        run(config, generator=blocked_generator)
    assert blocked_generator.prompts == []

    explicit_generator = RecordingBatchGenerator()
    run(replace(config, overwrite=True), generator=explicit_generator)
    assert explicit_generator.prompts == ["changed q0", "changed q1", "changed q2"]


def test_stale_current_fingerprint_shards_cannot_bypass_manifest_conflict(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("prompt-a {question}", encoding="utf-8")
    config = _resume_config(tmp_path, prompt_files={"vqa": prompt_path})
    run(config, generator=RecordingBatchGenerator())

    prompt_path.write_text("prompt-b {question}", encoding="utf-8")
    run(replace(config, overwrite=True), generator=RecordingBatchGenerator())

    prompt_path.write_text("prompt-a {question}", encoding="utf-8")
    blocked_generator = RecordingBatchGenerator()
    with pytest.raises(RuntimeError, match="--overwrite"):
        run(config, generator=blocked_generator)

    assert blocked_generator.prompts == []


def test_interrupted_explicit_overwrite_can_resume_without_clearing_new_shards(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("prompt-a {question}", encoding="utf-8")
    config = _resume_config(tmp_path, prompt_files={"vqa": prompt_path})
    run(config, generator=RecordingBatchGenerator())

    prompt_path.write_text("prompt-b {question}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(replace(config, overwrite=True), generator=FailingSecondBatchGenerator())

    resumed = RecordingBatchGenerator()
    run(config, generator=resumed)

    assert resumed.prompts == ["prompt-b q1", "prompt-b q2"]


@pytest.mark.parametrize(
    ("indexes", "message"),
    [(["", "2"], "empty index"), (["1", "1"], "duplicate index")],
)
def test_invalid_indexes_fail_before_model_construction(tmp_path, monkeypatch, indexes, message):
    config = _resume_config(tmp_path)
    pd.DataFrame(
        [
            {"index": index, "image": "", "question": f"q{position}", "answer": "a"}
            for position, index in enumerate(indexes)
        ]
    ).to_csv(config.tsv_dir / "aero_vqa.tsv", sep="\t", index=False)
    constructed = []

    def construct_generator(**kwargs):
        constructed.append(kwargs)
        return RecordingBatchGenerator()

    monkeypatch.setattr("eval_tool.run_infer.QwenVLGenerator", construct_generator)

    with pytest.raises(RuntimeError, match=message):
        run(config)
    assert constructed == []


def test_matching_manifest_skips_without_partial_shards_or_model_load(tmp_path, monkeypatch):
    config = _resume_config(tmp_path)
    written = run(config, generator=RecordingBatchGenerator())
    for shard in (config.out_dir / "_partial").glob("*.jsonl"):
        shard.unlink()

    constructed = []

    def construct_generator(**kwargs):
        constructed.append(kwargs)
        return RecordingBatchGenerator()

    monkeypatch.setattr("eval_tool.run_infer.QwenVLGenerator", construct_generator)

    reused = run(config)

    assert reused == written
    assert constructed == []


def test_xlsx_write_failure_preserves_previous_final(tmp_path, monkeypatch):
    config = _resume_config(tmp_path)
    output_path = run(config, generator=RecordingBatchGenerator())["vqa"]
    previous_bytes = output_path.read_bytes()

    def fail_after_partial_write(self, excel_writer, *args, **kwargs):
        Path(excel_writer).write_bytes(b"incomplete workbook")
        raise OSError("simulated xlsx failure")

    monkeypatch.setattr(pd.DataFrame, "to_excel", fail_after_partial_write)

    with pytest.raises(OSError, match="simulated xlsx failure"):
        run(replace(config, overwrite=True), generator=RecordingBatchGenerator())

    assert output_path.read_bytes() == previous_bytes


def test_manifest_write_failure_preserves_previous_final(tmp_path, monkeypatch):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("old {question}", encoding="utf-8")
    config = _resume_config(tmp_path, prompt_files={"vqa": prompt_path})
    output_path = run(config, generator=RecordingBatchGenerator())["vqa"]
    previous_bytes = output_path.read_bytes()
    final_manifest_path = output_path.with_suffix(".infer.json")
    prompt_path.write_text("new {question}", encoding="utf-8")

    def fail_final_manifest(path, manifest):
        if Path(path) == final_manifest_path:
            raise OSError("simulated manifest failure")
        write_manifest_atomic(path, manifest)

    monkeypatch.setattr("eval_tool.run_infer.write_manifest", fail_final_manifest)

    with pytest.raises(OSError, match="simulated manifest failure"):
        run(replace(config, overwrite=True), generator=RecordingBatchGenerator())

    assert output_path.read_bytes() == previous_bytes


def test_clean_partial_runs_only_after_successful_publication(tmp_path):
    config = _resume_config(tmp_path, clean_partial=True)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(config, generator=FailingSecondBatchGenerator())
    assert list((config.out_dir / "_partial").glob("vqa__*.jsonl"))

    run(config, generator=RecordingBatchGenerator())

    assert list((config.out_dir / "_partial").glob("vqa__*.jsonl")) == []


def test_dataset_lock_rejects_concurrent_resume_before_generation(tmp_path):
    config = _resume_config(tmp_path)
    lock_path = config.out_dir / "_partial" / "vqa.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("pid=other\n", encoding="utf-8")
    generator = RecordingBatchGenerator()

    with pytest.raises(RuntimeError, match="already running"):
        run(config, generator=generator)

    assert generator.prompts == []
    assert lock_path.is_file()


def test_resume_failure_is_appended_to_utf8_warnings_log(tmp_path):
    config = _resume_config(tmp_path)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(config, generator=FailingSecondBatchGenerator())

    warning_text = (config.out_dir / "warnings.log").read_text(encoding="utf-8")
    assert "aero_vqa" in warning_text
    assert "simulated interruption" in warning_text


def test_dataset_load_failure_is_appended_to_warnings_log(tmp_path):
    config = InferConfig(
        model_name="base",
        model_path=tmp_path / "model",
        tsv_dir=tmp_path / "missing-tsv",
        out_dir=tmp_path / "pred",
        datasets={"vqa": "missing_dataset"},
        prompt_files={},
        resume=True,
    )

    with pytest.raises(FileNotFoundError):
        run(config, generator=RecordingBatchGenerator())

    warning_text = (config.out_dir / "warnings.log").read_text(encoding="utf-8")
    assert "missing_dataset" in warning_text
    assert "FileNotFoundError" in warning_text


def test_legacy_xlsx_without_manifest_requires_pred_or_overwrite(tmp_path):
    config = _resume_config(tmp_path)
    config.out_dir.mkdir()
    legacy_path = config.out_dir / "base_aero_vqa.xlsx"
    pd.DataFrame([{"index": "0", "prediction": "legacy"}]).to_excel(
        legacy_path, index=False
    )
    generator = RecordingBatchGenerator()

    with pytest.raises(RuntimeError, match=r"declare.*pred.*--overwrite"):
        run(config, generator=generator)

    assert generator.prompts == []


def test_resume_rejects_unexpected_cached_index_before_generation(tmp_path):
    config = _resume_config(tmp_path)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run(config, generator=FailingSecondBatchGenerator())
    shard = next((config.out_dir / "_partial").glob("vqa__*.jsonl"))
    with shard.open("a", encoding="utf-8") as stream:
        stream.write('{"index":"unexpected","prediction":"stale"}\n')
    generator = RecordingBatchGenerator()

    with pytest.raises(RuntimeError, match="unexpected cached indexes"):
        run(config, generator=generator)

    assert generator.prompts == []


def test_manifest_metadata_mismatch_requires_overwrite_before_generation(tmp_path):
    config = _resume_config(tmp_path)
    output_path = run(config, generator=RecordingBatchGenerator())["vqa"]
    for shard in (config.out_dir / "_partial").glob("*.jsonl"):
        shard.unlink()
    manifest_path = output_path.with_suffix(".infer.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    generator = RecordingBatchGenerator()

    with pytest.raises(RuntimeError, match="--overwrite"):
        run(config, generator=generator)

    assert generator.prompts == []
