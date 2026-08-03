# 独立多模型 VLM 评估打分工具

## 统一入口（推荐）

复制 `pipeline.example.json` 为 `pipeline.json`，在一份配置里填写数据、多个模型、推理和裁判参数：

```bash
python -m eval_tool convert data.json --config pipeline.json
python -m eval_tool infer --config pipeline.json
python -m eval_tool eval --config pipeline.json --rubric v4
python -m eval_tool sweep --config pipeline.json --rubrics v1,v3,v4,v4b
python -m eval_tool all --config pipeline.json
```

- 推理默认断点续传；只有显式 `--overwrite` 才会从头跑。推理指纹变化时会报错，不会自动触发几小时的 GPU 重跑。
- 每个 batch 写入 `work_dir/<model>/_partial/*.jsonl`；默认保留以便核查，完成后可用 `--clean-partial` 清理。
- 评估输入优先级为 `models[].scored > models[].pred > 约定推理路径`。`sweep` 为了保证 rubric 可验证，拒绝使用 `scored`。
- 多模型和多 rubric 按配置顺序串行。`sweep` 的比较表会覆盖每个 challenger，不只是第一个。

旧入口仍保留：`python -m eval_tool.run_infer --config infer_config.json`、`python -m eval_tool.run_eval --config config.json`，以及无子命令的 `python -m eval_tool --config config.json`。

> **评估装备描述开放问答(ShareGPT JSON 数据、只跑 vqa)请看《开放问答评估_使用说明.md》**,那是当前主用通路。本 README 描述的是原有 mcq/judge/vqa 三数据集流程,部分提示词/配置示例(P1-R3 能力分类等)不适用于开放问答通路。

这是一个两段式 Python 命令行工具：

1. `run_infer`：读取航空评估集 TSV 和本地 HuggingFace Qwen-VL 权重目录，生成可复用的模型预测 xlsx。
2. `run_eval`：读取真值 TSV 和已有预测 xlsx/csv/tsv，产出多模型对比报告。

评估阶段不重新推理。你可以反复改裁判 prompt、统计逻辑或报告逻辑，然后直接复用已有预测文件重跑评估。

当前首版重点做扎实默认主链路：

- MCQ / 判断题：本地抽取选项字母并精确匹配。
- 开放题点式评分：调用看图裁判模型，结果写入 JSONL 缓存，重跑跳过已判条目。
- 开放题成对比较：每个模型 vs `BASELINE_MODEL`，正反两个方向裁判后合并，消除位置偏差。
- 报告：总表、长表、交叉表、逐条明细、成对汇总、成对明细、warnings。
- 统计：bootstrap 置信区间，开放题点式长度控制分，成对胜率长度控制回退/估计。

暂不实现全量 `round_robin` 和 Elo/Bradley-Terry 排名。

## 运行

### 1. 推理生成可复用 xlsx

先复制推理配置并改路径：

```powershell
Copy-Item infer_config.example.json infer_config.json
```

运行：

```powershell
python -m eval_tool.run_infer --config infer_config.json
```

输出示例：

```text
F:/path/to/work_dir/base/base_aero_mcq.xlsx
F:/path/to/work_dir/base/base_aero_judge.xlsx
F:/path/to/work_dir/base/base_aero_vqa.xlsx
```

推理脚本直接加载 HuggingFace 本地权重目录，例如 Qwen2.5-VL / Qwen3-VL。需要你的 Python 环境已安装对应版本的 `torch`、`transformers`、`Pillow` 等依赖。

推理性能相关配置：

```json
{
  "batch_size": 1,
  "device_map": "auto",
  "gpu_ids": [],
  "workers_per_gpu": 1
}
```

- `batch_size`：单个模型实例一次处理多少条。显存够可以调大，比如 2、4、8。
- `device_map: "auto"`：让 transformers/accelerate 自动把一个大模型切到多张卡上，这是模型并行，适合单卡放不下模型。
- `gpu_ids: [0, 1, 2, 3]`：启用数据并行。脚本会为每张卡启动独立 worker，每个 worker 加载一份模型，分片处理 TSV 行，最后按原 index 顺序合并输出。
- `workers_per_gpu`：每张卡几个 worker。通常先用 1；同卡多 worker 只有在模型小、显存足且 GPU 利用率低时才考虑。

两种多卡方式不要混着理解：`device_map=auto` 是一个模型跨多卡；`gpu_ids` 是多份模型多进程跑不同数据。一般推荐：

- 模型单卡能放下：用 `gpu_ids: [0,1,...]` 做数据并行。
- 模型单卡放不下：先用 `device_map: "auto"`，不要设置 `gpu_ids`。

### 2. 复用 xlsx 做评估

先复制示例配置并改路径：

```powershell
Copy-Item config.example.json config.json
```

运行：

```powershell
python -m eval_tool --config config.json
```

### 只评估部分数据集 / 六维度加权

`config.json` 里可以加两个可选字段：

```json
{
  "enabled_datasets": ["vqa"],
  "category_weights": {
    "P1": 1.0, "P2": 1.0, "P3": 1.0,
    "R1": 0.2, "R2": 0.2, "R3": 0.2
  }
}
```

- `enabled_datasets`：只写你要跑的数据集 key（`mcq` / `judge` / `vqa` 的子集）。不在列表里的数据集完全不加载真值 TSV，也不要求模型配置对应的预测路径。默认是三个都跑。
- `category_weights`：六个能力维度（P1/P2/P3/R1/R2/R3）的加权系数，用于计算 `score_summary.csv` 里的 `total_score`。三个数据集（mcq/judge/vqa）的样本会先按 `category` 汇总成每个维度各自的原始分，再按这套权重加权平均得到总分。缺省全部是 1.0；没有对应训练任务的维度（比如目前的 R1/R2/R3）可以调低权重，之后有训练任务了再调回去。不影响 `report_summary.csv` 等其他文件，只影响 `score_summary.csv` 的 `total_score`。

## Prompt 文件

推理 prompt 和裁判 prompt 都放在外部文件里：

```text
prompts/
  infer_mcq.txt
  infer_judge.txt
  infer_vqa.txt
  judge_vqa_pointwise.txt
  judge_vqa_pairwise.txt
```

推理 prompt 支持用 `{question}`、`{A}`、`{B}`、`{C}`、`{D}`、`{answer}`、`{category}`、`{l2-category}`、`{source_id}` 等 TSV 列名占位。改推理 prompt 后需要重跑 `run_infer`；改裁判 prompt 后只需要重跑 `run_eval`。

## 输入

真值目录里需要有：

- `aero_mcq.tsv`
- `aero_judge.tsv`
- `aero_vqa.tsv`

预测文件每个模型每个数据集一份，至少包含：

- `index`
- `prediction`，也兼容 `pred`、`response`、`model_answer`、`answer_pred`、`模型回答`、`预测`

## 输出

默认写到配置里的 `out_dir`：

- `score_summary.csv`：**精简总表**，每个模型一行，列为 `total_score`（按 `category_weights` 加权）+ 各能力维度（P1/P2/P3/R1/R2/R3）原始分及样本数。运行结束会直接打印在终端。
- `report_summary.csv` / `report_summary.json`：模型横向总表，含 CI 和长度控制列。
- `report_summary_long.csv`：长格式指标表。
- `cross_{model}.csv`：能力 × 内容类型交叉表，含分数和样本数。
- `detail_{model}_{dataset}.xlsx`：逐条明细，不写入 base64 图片。
- `pairwise_vs_baseline.csv`：各模型 vs 基准的胜/平/负率。
- `pairwise_vs_baseline_detail.xlsx`：成对逐条明细。
- `warnings.log`：缺预测、额外 index、跳过项等警告。

缓存默认写到 `cache_dir`：

- `judge_cache_pointwise.jsonl`
- `judge_cache_pairwise.jsonl`

缓存不要删；重跑和加模型会复用它。
