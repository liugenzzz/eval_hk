# DPO Shard Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从完整 inference 分片安全生成可直接训练的 DPO JSONL，并改善零 pending 恢复阶段的可见性和扫描效率。

**Architecture:** 根目录单文件脚本复用现有配置和输入归一化模块，但自行读取、校验和汇合分片，不进入模型/多进程流水线。项目内仅对 `run_dpo_inference` 做局部缓存复用和状态输出。

**Tech Stack:** Python 3.10+、标准库 JSON/argparse、tqdm、pytest。

---

### Task 1: 独立分片恢复脚本

**Files:**
- Create: `merge_dpo_shards.py`
- Create: `tests/test_merge_dpo_shards.py`

- [ ] 先写测试，构造两个原始样本和完整 worker shard，断言严格四字段输出及原始顺序。
- [ ] 运行测试，确认因模块不存在而失败。
- [ ] 实现 active attempt 定位、严格 shard 校验、候选映射、过滤和原子 JSONL 输出。
- [ ] 增加重复 sample、记录数不符和 `wrong_only=true` 的失败测试。
- [ ] 运行 `python -m pytest tests/test_merge_dpo_shards.py -q`，确认通过。

### Task 2: 零 pending 恢复优化

**Files:**
- Modify: `eval_tool/dpo_infer.py`
- Modify: `tests/test_dpo_infer.py`

- [ ] 写测试证明首次 `store.load()` 的结果用于计算 pending，不再立即调用 `pending_sample_ids()` 二次扫描。
- [ ] 运行测试确认失败。
- [ ] 用已加载 key 计算 pending，并输出缓存命中、待推理和零 pending 发布提示。
- [ ] 运行 DPO inference 与 pipeline 测试。

### Task 3: 交付验证

- [ ] 用合成分片实际运行独立脚本并逐行校验输出 schema。
- [ ] 运行 `python -m py_compile merge_dpo_shards.py eval_tool/dpo_infer.py`。
- [ ] 打包脚本、修复文件和一页运行说明，核对压缩包内容与哈希。
