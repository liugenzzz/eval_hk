# DPO Parallel Progress Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely report aggregate multi-worker DPO inference progress without blocking worker shutdown or artifact publication.

**Architecture:** Workers append durable inference outcomes first and then send one `progress` message per completed batch through the existing status queue. The parent owns one tqdm instance inside `_wait_for_worker_completion`, validates every increment, and closes it in `finally` on every exit path.

**Tech Stack:** Python 3.12, multiprocessing, tqdm, pytest

---

### Task 1: Specify parent-side progress behavior

**Files:**
- Modify: `tests/test_dpo_infer.py`
- Modify: `eval_tool/dpo_infer.py`

- [ ] **Step 1: Write a failing test for successful progress and close**

Add a fake status queue with `progress` then `done`, monkeypatch `eval_tool.dpo_infer.tqdm`, call `_wait_for_worker_completion(..., total_batches=1)`, and assert one update plus one close.

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_dpo_infer.py::test_worker_completion_updates_and_closes_parent_progress -q --basetemp .pytest_progress_red`

Expected: FAIL because `_wait_for_worker_completion` does not accept `total_batches` and `progress` is not an allowed message.

- [ ] **Step 3: Implement the minimal parent-side protocol**

Import `tqdm`, add `progress` to `_next_worker_message`, pass `total_batches` from `_run_parallel_inference`, validate increments, and close the parent progress bar in `finally`.

- [ ] **Step 4: Run the test and verify GREEN**

Run: `python -m pytest tests/test_dpo_infer.py::test_worker_completion_updates_and_closes_parent_progress -q --basetemp .pytest_progress_green`

Expected: PASS.

### Task 2: Report durable worker progress and cover failure cleanup

**Files:**
- Modify: `tests/test_dpo_infer.py`
- Modify: `eval_tool/dpo_infer.py`

- [ ] **Step 1: Write a failing test for fatal-path close**

Feed a `fatal` status to `_wait_for_worker_completion`, assert `DpoInferenceFatalError`, and assert the fake tqdm instance closes exactly once.

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_dpo_infer.py::test_worker_completion_closes_parent_progress_on_fatal -q --basetemp .pytest_progress_fatal_red`

Expected: FAIL until unconditional close is implemented.

- [ ] **Step 3: Send progress after each durable worker batch**

After `_infer_batch_with_isolation` returns inside `_dpo_inference_worker_entry`, send `("progress", worker_id, 1, None)`. Keep `done` as the terminal message after all assigned batches.

- [ ] **Step 4: Run focused and full verification**

Run: `python -m pytest tests/test_dpo_infer.py -q --basetemp .pytest_progress_infer`

Run: `python -m pytest -q --basetemp .pytest_progress_all`

Expected: all tests PASS and no multiprocessing shutdown error occurs.
