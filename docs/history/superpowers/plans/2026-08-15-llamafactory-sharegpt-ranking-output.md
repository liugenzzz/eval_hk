# LLaMAFactory ShareGPT Ranking Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every generated DPO JSONL use LLaMAFactory-compatible ShareGPT pairwise messages for `chosen` and `rejected`.

**Architecture:** Keep the existing normalized `conversations` and `images` fields unchanged. Wrap the two answer strings at the two final serialization boundaries, then make staged-artifact validation enforce the same contract so invalid legacy output cannot be published.

**Tech Stack:** Python 3.12, JSONL, pytest, LLaMAFactory ShareGPT ranking contract.

---

### Task 1: Main pipeline output

**Files:**
- Modify: `tests/test_dpo_pipeline.py`
- Modify: `eval_tool/dpo_pipeline.py:513-522`

- [ ] Change `test_short_valid_answers_are_not_length_filtered` to require:

```python
"chosen": {"from": "gpt", "value": "是"},
"rejected": {"from": "gpt", "value": "否"},
```

- [ ] Run the single test and confirm it fails because the current output contains strings.
- [ ] Wrap `candidate.chosen` and `prediction` in canonical `gpt` messages in `_training_row`.
- [ ] Re-run the single test and confirm it passes.

### Task 2: Standalone shard merger output

**Files:**
- Modify: `tests/test_merge_dpo_shards.py`
- Modify: `merge_dpo_shards.py:236-245`

- [ ] Change the recovery test to require `gpt` message objects for both answer fields.
- [ ] Run the recovery test and confirm it fails against the string output.
- [ ] Apply the same answer wrapping in the merger's `_training_row`.
- [ ] Re-run the recovery test and confirm it passes while image paths remain unchanged.

### Task 3: Artifact validation

**Files:**
- Modify: `tests/test_dpo_report.py`
- Modify: `eval_tool/dpo_report.py:795-820`

- [ ] Make the valid test fixture use ShareGPT answer messages and add a case that rejects legacy strings.
- [ ] Run the focused report tests and confirm the current string-only validator fails the new valid fixture.
- [ ] Validate that each answer is exactly `{"from": "gpt", "value": <string>}`.
- [ ] Re-run the focused report tests and confirm valid messages pass and legacy strings fail.

### Task 4: Regression expectations and documentation

**Files:**
- Modify: `tests/test_dpo_pipeline.py`
- Modify: `tests/test_dpo_pipeline_resume.py`
- Modify: `DPO_使用说明.md`
- Modify: `DPO_分片直接合并_使用说明.md`
- Modify: `README_DPO_SHARD_MERGE.md`

- [ ] Update remaining answer assertions to read the message `value` while retaining exact ordering checks.
- [ ] Replace old JSON examples with canonical ShareGPT ranking objects.
- [ ] Add the required LLaMAFactory `dataset_info.json` mapping with `formatting: sharegpt` and `ranking: true`.
- [ ] Search the maintained source, tests, and DPO documentation for legacy string examples.

### Task 5: Verification and delivery

**Files:**
- Verify all modified source, tests, and docs.

- [ ] Run focused DPO pipeline, report, resume, and shard-merger tests.
- [ ] Run `python -m py_compile` on modified Python modules.
- [ ] Run the full pytest suite.
- [ ] Build a fresh project archive excluding caches, generated archives, and Git metadata; compute SHA-256.

The shared worktree already contains requested uncommitted changes, so this execution intentionally avoids Git commits and preserves the existing history.
