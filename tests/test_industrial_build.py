"""Full frozen build with mocked private transport and real JSONL/images/YAML."""
import json
from pathlib import Path

import pytest
from PIL import Image

from industrial_build import build_quality_dataset


class Reviewer:
    cache_settings = {"provider": "fixture", "model": "fixture-v1"}

    def __init__(self):
        self.calls = 0

    def review(self, prompt, image_paths=()):
        self.calls += 1
        payload = json.loads(prompt.split("INPUT:\n", 1)[1])
        if "entry_id" in payload:
            candidates = payload.get("canonical_candidates", [])
            canonical = candidates[0] if candidates else "crack"
            verdict = {"entry_id": payload["entry_id"], "decision": "accepted",
                       "canonical": canonical,
                       "defect_status": ("normal" if canonical in {"normal", "good"} else "abnormal")
                                        if payload["kind"] == "defect" else None,
                       "reason": "fixture semantic mapping check"}
        else:
            assert image_paths and all(Path(p).exists() for p in image_paths)
            verdict = {"decision": "usable", "reason": "fixture image/reference consistency check"}
        return {"content": json.dumps(verdict), "provider": "fixture", "model": "fixture-v1"}


def source_files(root):
    image_dir = root / "defect-detect-images" / "fixture"
    image_dir.mkdir(parents=True)
    records = {"zh": [], "en": []}
    for obj, zh_obj in (("bottle", "瓶子"), ("cable", "电缆")):
        for defect, zh_defect in (("normal", "正常"), ("crack", "裂纹")):
            for i in range(2):
                image = image_dir / f"{obj}_{defect}_{i}.png"
                Image.new("RGB", (8, 8), (len(records['zh']) + 1, 120, 180)).save(image)
                for lang in records:
                    answer = (f"Category: {obj}; Anomaly type: {defect}; Description: fixture"
                              if lang == "en" else f"类别：{zh_obj}；异常类型：{zh_defect}；描述：测试")
                    records[lang].append({"images": [str(image)], "conversations": [
                        {"from": "human", "value": "<image>describe"},
                        {"from": "gpt", "value": answer}]})
    paths = {}
    for lang, data in records.items():
        paths[lang] = root / f"{lang}.jsonl"
        paths[lang].write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data), encoding="utf-8")
    return paths


def test_entire_build_and_resume_freeze_mapping_and_enforce_coverage(tmp_path):
    import yaml
    from eval_tool.config import load_pipeline_config
    from eval_tool.io import load_truth_dataset

    inputs = source_files(tmp_path)
    client = Reviewer()
    output = tmp_path / "build"
    result = build_quality_dataset(**inputs, out_dir=output, zh_n=6, en_n=6,
                                   object_min=2, client=client)
    assert result.is_file()
    frozen = yaml.safe_load((output / "industrial_labels.yaml").read_text(encoding="utf-8"))
    assert frozen["status"] == "frozen"
    rows = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len({r["id"] for r in rows}) == 12
    assert {r["metadata"]["defect_status"] for r in rows} == {"normal", "abnormal"}
    report = json.loads((output / "evaluation/coverage_report.json").read_text(encoding="utf-8"))
    assert not report["blockers"]
    assert all(cell["gap"] == 0 for cell in report["coverage"])
    config = load_pipeline_config(Path(__file__).resolve().parents[1] / "industrial_mixed.example.json")
    for lang in ("zh", "en"):
        params = {**config.dataset_params[f"industrial_{lang}"], "image_root": str(tmp_path)}
        frame = load_truth_dataset(result.parent, "industrial_eval", params)
        assert len(frame) == 6
        assert frame["image"].str.len().min() > 0
    before = client.calls
    resumed = build_quality_dataset(**inputs, out_dir=output, zh_n=6, en_n=6,
                                    object_min=2, client=client, resume=True)
    assert resumed == result
    assert client.calls == before
    image = Path(rows[0]['images'][0])
    Image.new('RGB', (8, 8), 'black').save(image)
    with pytest.raises(ValueError, match="image changed"):
        build_quality_dataset(**inputs, out_dir=output, zh_n=6, en_n=6,
                              object_min=2, client=client, resume=True)
    with pytest.raises(FileExistsError):
        build_quality_dataset(**inputs, out_dir=output, zh_n=6, en_n=6, object_min=2, client=client)


def test_budget_failure_cannot_emit_a_half_complete_dataset(tmp_path):
    inputs = source_files(tmp_path)
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="[Cc]overage|budget|capacity"):
        build_quality_dataset(**inputs, out_dir=output, zh_n=2, en_n=2, client=Reviewer())
    assert not (output / "evaluation/industrial_eval.jsonl").exists()
    assert (output / "plan/coverage_plan.json").exists()


def test_incomplete_mapping_review_stops_before_sample_judging(tmp_path):
    inputs = source_files(tmp_path)
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="missing review"):
        build_quality_dataset(**inputs, out_dir=output, max_mapping_reviews=0, client=Reviewer())
    assert not (output / "sample_reviews.jsonl").exists()
    assert not (output / "industrial_labels.yaml").exists()


def test_mapping_review_errors_are_skipped_during_sampling(tmp_path):
    class FailingReviewer(Reviewer):
        def review(self, prompt, image_paths=()):
            payload = json.loads(prompt.split("INPUT:\n", 1)[1])
            if payload.get("kind") == "object" and payload.get("alias") == "瓶子":
                raise RuntimeError("fixture provider failure")
            return super().review(prompt, image_paths)

    inputs = source_files(tmp_path)
    output = tmp_path / "build"
    result = build_quality_dataset(**inputs, out_dir=output, zh_n=2, en_n=4,
                                   client=FailingReviewer())
    assert result.is_file()
    audit = [json.loads(line) for line in
             (output / "evaluation/quality_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(row.get("reason") == "unresolved_mapping:unresolved" for row in audit)


def test_zero_sample_review_budget_builds_with_structural_checks(tmp_path):
    inputs = source_files(tmp_path)
    output = tmp_path / "build"
    client = Reviewer()
    result = build_quality_dataset(**inputs, out_dir=output, zh_n=4, en_n=4,
                                   max_sample_reviews=0, client=client)
    assert result.is_file()
    assert not (output / "sample_reviews.jsonl").exists()
    report = json.loads((output / "evaluation/coverage_report.json").read_text(encoding="utf-8"))
    assert report["quality_checked"] == "structural"


def test_resume_can_change_targets_before_sampling(tmp_path):
    inputs = source_files(tmp_path)
    output = tmp_path / "build"
    with pytest.raises(ValueError, match="missing review"):
        build_quality_dataset(**inputs, out_dir=output, zh_n=6, en_n=6,
                              max_mapping_reviews=0, client=Reviewer())
    result = build_quality_dataset(**inputs, out_dir=output, zh_n=4, en_n=4,
                                   max_sample_reviews=0, resume=True, client=Reviewer())
    assert result.is_file()


def test_limited_audit_can_resume_without_overwriting_failure_reports(tmp_path):
    inputs = source_files(tmp_path)
    output = tmp_path / 'build'
    client = Reviewer()
    with pytest.raises(ValueError, match='Coverage blocked'):
        build_quality_dataset(**inputs, out_dir=output, zh_n=6, en_n=6,
                              max_sample_reviews=2, client=client)
    assert (output / 'plan/coverage_report.json').exists()
    dataset = build_quality_dataset(**inputs, out_dir=output, zh_n=6, en_n=6,
                                    resume=True, client=client)
    assert dataset.is_file()
    assert (output / 'plan_attempt_2/coverage_plan.json').exists()


def test_frozen_validation_blocks_inference_and_scoring_before_calls(tmp_path):
    from dataclasses import replace
    from eval_tool.config import load_pipeline_config
    from eval_tool.run_infer import run as infer
    from eval_tool.run_eval import run as evaluate

    inputs = source_files(tmp_path)
    dataset = build_quality_dataset(**inputs, out_dir=tmp_path/'build', zh_n=4, en_n=4, client=Reviewer())
    config = load_pipeline_config(Path(__file__).resolve().parents[1]/'industrial_quality.example.json')
    # Valid data is bound by manifest; editing even whitespace is detected before GPU/API work.
    with dataset.open('a', encoding='utf-8') as stream:
        stream.write('\n')
    ic = replace(config.to_infer_configs(['base'])[0], tsv_dir=dataset.parent)
    ec = replace(config.to_eval_config(), tsv_dir=dataset.parent)
    with pytest.raises(ValueError, match='JSONL changed'):
        infer(ic)
    with pytest.raises(ValueError, match='JSONL changed'):
        evaluate(ec)


def test_relative_training_images_work_without_root_from_another_cwd(tmp_path, monkeypatch):
    inputs = source_files(tmp_path)
    for path in inputs.values():
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        for row in rows:
            row['images'] = [str(Path(row['images'][0]).relative_to(path.parent))]
        path.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    result = build_quality_dataset(**inputs, out_dir=tmp_path/'build', zh_n=4, en_n=4, client=Reviewer())
    output = [json.loads(line) for line in result.read_text(encoding='utf-8').splitlines()]
    assert len(output) == 8
    assert all(Path(row['images'][0]).is_absolute() for row in output)
    assert all(Path(row['images'][0]).is_file() for row in output)
    assert all(not Path(row['metadata']['original_images'][0]).is_absolute() for row in output)
