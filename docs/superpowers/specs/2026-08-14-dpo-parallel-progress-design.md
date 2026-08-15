# DPO 并行推理总进度修复设计

## 目标

保留多 GPU、多 worker 推理时的单个 `tqdm` 总进度条，同时保证进度显示不会影响分片持久化、worker 终止检测、缓存完成标记和最终产物发布。

## 设计

- 每个 worker 仅在一个 batch 的全部终态记录已经写入分片后，向现有 `status_queue` 发送四元组 `("progress", worker_id, 1, None)`。
- 父进程在启动 worker 前根据 `launch.batches` 计算 `total_batches`，并传给 `_wait_for_worker_completion`。
- `_next_worker_message` 接受 `progress` 类型，但 `_wait_for_worker_completion` 负责校验增量必须是正整数、累计值不得超过 `total_batches`。
- `tqdm` 只在 `_wait_for_worker_completion` 内创建和使用，并通过 `try/finally` 在成功、worker 异常和协议异常时统一关闭；禁止在调用方引用 `pbar`。
- 全部 worker 正常发出 `done` 后，累计进度必须等于 `total_batches`，否则按协议错误终止，避免显示完成但缓存不完整。
- 没有待推理样本时不创建进度条，现有断点缓存和发布流程保持不变。

## 错误处理

- 非法、负数、零或超量的 progress 消息触发 `DpoInferenceFatalError`。
- worker fatal、peer stop 或进度协议错误都必须关闭进度条，再由现有清理逻辑终止子进程。
- 不更改缓存 fingerprint、JSONL 格式、attempt 目录或最终 DPO 文件格式，因此已有分片可以直接续用。

## 验证

- 单元测试验证 progress 更新总进度并在正常退出时关闭。
- 单元测试验证 fatal 消息出现时进度条仍关闭。
- 现有 spawn 并行测试验证真实 worker 可以上报进度、退出并完成缓存。
- 运行 `tests/test_dpo_infer.py`，再运行完整测试集。
