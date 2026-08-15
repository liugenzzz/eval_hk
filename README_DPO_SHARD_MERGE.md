# DPO shard merge

`merge_dpo_shards.py` directly builds a trainable DPO JSONL from the original input JSONL files and completed inference shards.

It reads `inputs`, `output_dir`, `output_name`, and `work_dir` from `dpo_full.json`. Alpaca and ShareGPT inputs are normalized with the project's existing rules. The original answer becomes `chosen`, the matching shard prediction becomes `rejected`, and every `image` or `images` path is preserved exactly in the output `images` field. Both preference answers are emitted as ShareGPT assistant messages (`{"from":"gpt","value":"..."}`), as required by LLaMAFactory with `formatting: sharegpt` and `ranking: true`.

The script does not load a model, use a GPU, call the Judge, or filter predictions by correctness. `wrong_only=true` is intentionally ignored.

Stop the existing builder before running this standalone recovery script:

```bash
pkill -f 'python -m eval_tool build-dpo' || true
cd /code/eval_vqa_v1
python merge_dpo_shards.py --config dpo_full.json
```

With the current config, the default output is:

```text
/code/data_process/dpo/output/train_dpo_base.recovered.jsonl
```

To choose a path explicitly:

```bash
python merge_dpo_shards.py \
  --config dpo_full.json \
  --output /code/data_process/dpo/output/train_dpo_base.jsonl
```

Use `--force` only when an existing destination is known to be safe to replace.
