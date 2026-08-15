# DPO 分片直接恢复设计

目标是绕开当前卡住的 `build-dpo` 封装，从已经完成的 inference `worker-*.jsonl` 分片直接生成可训练 JSONL。

实现为项目根目录的单文件脚本 `merge_dpo_shards.py`，不注册 CLI、不加载模型、不启动子进程。脚本读取 `dpo_full.json`，通过 `work_dir/inference/active.json` 找到当前 attempt，严格校验分片 JSON、字段、重复 `sample_id` 和总记录数；再按原始输入顺序重建候选，将成功且非空、与 chosen 不同的 prediction 写成严格四字段训练行：`conversations`、`chosen`、`rejected`、`images`。

脚本明确执行“全量、不评估”恢复：无论配置中的 `wrong_only` 是什么，都不读取或调用 Judge，而是将所有有效推理结果作为 `rejected`。默认输出为 `output_dir/<output_stem>.recovered.jsonl`，通过临时文件原子替换，并打印读取、汇合进度和最终统计。

项目内后续修复只处理已确认的问题：复用第一次缓存扫描结果，打印缓存命中/待推理数量，并在零 pending 时明确显示“正在从缓存完成发布”，避免重复扫描阶段看起来像死锁。
