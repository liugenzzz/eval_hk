"""Build a frozen industrial evaluation set with model review and coverage gates.

Use --resume to reuse completed stages. No judge or label model is called by the
subsequent eval_tool infer/eval commands against the frozen evaluation dataset
except the normal task-quality judge during eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_quality_dataset(
    zh, en, out_dir, *, zh_n=500, en_n=500, object_min=1, defect_min=1,
    status_min=1, seed=42, version="industrial_labels_v1", provider=None,
    image_root=None, resume=False, allow_partial_mapping=False,
    allow_partial_coverage=False, max_mapping_reviews=None, max_sample_reviews=None,
    client=None,
):
    """Run all stages; any evidence/coverage failure stops before final output."""
    from eval_tool.industrial_mapping import FrozenMapping, scan, freeze
    from eval_tool.industrial_review import mapping_review, sample_review
    from eval_tool.industrial_quality_sample import sample_quality

    output = Path(out_dir)
    inputs = {"zh": {"path": str(Path(zh).resolve()), "sha256": _sha(zh)},
              "en": {"path": str(Path(en).resolve()), "sha256": _sha(en)}}
    options = {"zh_n": zh_n, "en_n": en_n, "object_min": object_min,
               "defect_min": defect_min, "status_min": status_min, "seed": seed,
               "version": version, "provider": provider, "image_root": str(image_root) if image_root else None,
               "allow_partial_mapping": allow_partial_mapping,
               "allow_partial_coverage": allow_partial_coverage}
    prompt_dir = Path(__file__).resolve().parent / "prompts"
    options["review_prompt_hashes"] = {
        name: _sha(prompt_dir / name) for name in
        ("industrial_mapping_review_v1.txt", "industrial_sample_review_v1.txt")
    }
    state_path = output / "build_state.json"
    if state_path.exists():
        if not resume:
            raise FileExistsError(f"Use --resume or a new build directory: {output}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["sources"] != inputs:
            raise ValueError("Build sources/options changed; use a new version/output directory")
        if state["options"] != options:
            changed = {key for key in set(state["options"]) | set(options)
                       if state["options"].get(key) != options.get(key)}
            if changed <= {"zh_n", "en_n"} and not ({"plan", "sample"} & set(state["completed"])):
                state["options"] = options
                state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                raise ValueError("Build sources/options changed; use a new version/output directory")
        for relative, expected in state.get("artifacts", {}).items():
            path = output / relative
            if not path.is_file() or _sha(path) != expected:
                raise ValueError(f"Completed build artifact changed or missing: {relative}")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Build output must be empty: {output}")
        output.mkdir(parents=True, exist_ok=True)
        state = {"schema_version": 1, "sources": inputs, "options": options, "completed": []}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def mark(stage, artifacts=()):
        if stage not in state["completed"]:
            state["completed"].append(stage)
        for path in artifacts:
            state.setdefault("artifacts", {})[str(path.relative_to(output))] = _sha(path)
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[industrial-build] completed: {stage}", flush=True)

    candidate_dir = output / "mapping_scan"
    candidate = candidate_dir / "candidate.yaml"
    queue = candidate_dir / "review_queue.jsonl"
    reviews = output / "mapping_reviews.jsonl"
    frozen = output / "industrial_labels.yaml"
    quality_reviews = output / "sample_reviews.jsonl"
    if "scan" not in state["completed"]:
        scan(en, zh, candidate_dir, version=version)
        mark("scan", (candidate, queue, candidate_dir / "scan_report.json"))
    if not frozen.exists():
        print("[industrial-build] reviewing ambiguous labels/status with private models", flush=True)
        review_result = mapping_review(candidate, queue, reviews, max_items=max_mapping_reviews,
                                       resume=reviews.exists(), provider=provider, client=client)
        try:
            freeze(candidate, reviews, frozen, allow_partial=allow_partial_mapping)
        except ValueError as error:
            raise ValueError(
                f"mapping review incomplete: {error}; progress={review_result}; "
                f"inspect {reviews} for status=error and rerun with --resume"
            ) from error
    # Re-validate on resume; a candidate or unrelated file cannot replace the freeze.
    frozen_mapping = FrozenMapping.load(frozen)
    frozen_mapping.validate_sources(zh=zh, en=en)
    mark("mapping_frozen", (frozen,))
    structural_only = max_sample_reviews == 0
    if "sample" in state["completed"]:
        from eval_tool.industrial_frozen_check import verify_frozen_dataset

        result = output / "evaluation" / "industrial_eval.jsonl"
        verify_frozen_dataset(result)
        return result
    if "plan" not in state["completed"]:
        if structural_only:
            print("[industrial-build] skipping per-sample model review; using structural checks", flush=True)
        else:
            print("[industrial-build] checking image/question/reference quality", flush=True)
            sample_review(zh, en, frozen, quality_reviews, image_root=image_root,
                          max_items=max_sample_reviews, resume=quality_reviews.exists(),
                          provider=provider, client=client)
        mark("quality_review_attempted")
    common = dict(zh_n=zh_n, en_n=en_n, seed=seed, object_min=object_min,
                  defect_min=defect_min, status_min=status_min, image_root=image_root,
                  quality_review=None if structural_only else quality_reviews,
                  quality_mode="structural" if structural_only else "semantic",
                  allow_partial_coverage=allow_partial_coverage)
    plan_dir = output / state.get("plan_dir", "plan")
    if "plan" not in state["completed"]:
        # Keep failed coverage audits. New reviews may make a later attempt feasible.
        attempt = 1
        while plan_dir.exists() and any(plan_dir.iterdir()):
            attempt += 1
            plan_dir = output / f"plan_attempt_{attempt}"
        sample_quality(zh, en, frozen, plan_dir, plan_only=True, **common)
        state["plan_dir"] = str(plan_dir.relative_to(output))
        artifacts = [plan_dir / "coverage_plan.json"]
        if not structural_only:
            artifacts.append(quality_reviews)
        mark("plan", artifacts)
    eval_dir = output / "evaluation"
    if "sample" not in state["completed"]:
        sample_quality(zh, en, frozen, eval_dir, plan=plan_dir / "coverage_plan.json", **common)
        mark("sample", (eval_dir / "industrial_eval.jsonl", eval_dir / "manifest.json",
                        eval_dir / "coverage_report.json"))
    return eval_dir / "industrial_eval.jsonl"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("zh", "en", "out-dir"):
        parser.add_argument("--" + flag, required=True, type=Path)
    for flag, default in (("zh-n", 500), ("en-n", 500), ("object-min", 1),
                          ("defect-min", 1), ("status-min", 1), ("seed", 42)):
        parser.add_argument("--" + flag, type=int, default=default)
    parser.add_argument("--version", default="industrial_labels_v1")
    parser.add_argument("--provider")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-partial-mapping", action="store_true")
    parser.add_argument("--allow-partial-coverage", action="store_true")
    parser.add_argument("--max-mapping-reviews", type=int)
    parser.add_argument("--max-sample-reviews", type=int)
    args = parser.parse_args(argv)
    try:
        result = build_quality_dataset(**vars(args))
    except (ValueError, FileExistsError, OSError, RuntimeError) as error:
        parser.exit(2, f"industrial-build stopped: {error}\n")
    print(result)


if __name__ == "__main__":
    main()
