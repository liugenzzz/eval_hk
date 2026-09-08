# DPO 分片直接合并

该脚本只做一件事：把 `dpo_full.json` 中三个原始 JSONL 的问题、正确答案和图片路径，与 inference attempt 下的 `worker-*.jsonl` 模型回答按 `sample_id` 合并，生成可训练 DPO JSONL。

它不会加载模型、不会使用 GPU、不会调用 Judge，也不会根据回答正确或错误进行筛选。配置中的 `wrong_only=true` 会被忽略。

## 使用

先停止仍在运行的原项目命令，避免它同时写输出：

```bash
pkill -f 'python -m eval_tool build-dpo' || true
```

把 `merge_dpo_shards.py` 放到 `/code/eval_vqa_v1/`，然后在项目根目录执行：

```bash
cd /code/eval_vqa_v1
python merge_dpo_shards.py --config dpo_full.json
```

按当前配置，默认生成：

```text
/code/data_process/dpo/output/train_dpo_base.recovered.jsonl
```

如果该文件已经存在并确认需要替换：

```bash
python merge_dpo_shards.py --config dpo_full.json --force
```

也可以指定输出：

```bash
python merge_dpo_shards.py \
  --config dpo_full.json \
  --output /code/data_process/dpo/output/train_dpo_base.jsonl
```

## 输出格式

每行严格只有四个字段：

```json
{
  "conversations": [{"from": "human", "value": "<image>\n问题"}],
  "chosen": {"from": "gpt", "value": "原始 JSONL 中的正确答案"},
  "rejected": {"from": "gpt", "value": "worker 分片中的模型回答"},
  "images": ["原始 JSONL 的 images 路径"]
}
```

这是 LLaMAFactory 的 ShareGPT ranking 格式；`chosen/rejected` 不是裸字符串，
`images` 中的原始路径不会被改写。

脚本会校验分片记录数、JSON 格式、字段、重复 `sample_id` 和输入映射。任何一项不一致都会停止，不会留下半截训练文件。
