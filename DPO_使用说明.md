# DPO 数据构建功能使用说明

## 1. 功能说明

新增命令 `build-dpo` 可直接把 Alpaca 或 ShareGPT 格式的 JSON/JSONL 数据构造成 DPO 训练集：

- 原数据中的标准答案作为 `chosen`；
- 本地 Qwen-VL 兼容模型的生成结果作为 `rejected`；
- 可选调用 Judge，只保留模型确实答错的样本；
- 支持文本、单图、多图、多轮对话、断点续跑和原子发布；
- 最终输出严格的 ShareGPT DPO JSONL，不需要先转换为 TSV/XLSX。

命令入口：

```powershell
python -m eval_tool build-dpo --help
```

## 2. 环境准备

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
```

推理环境至少需要 `torch`、`transformers`、`accelerate` 和 `Pillow`。`transformers` 版本必须包含与所用 Qwen2.5-VL/Qwen3-VL checkpoint 兼容的模型类。模型文件需已下载到本地，配置中的 `infer.model_path` 指向该目录。

建议先确认命令可用：

```powershell
python -m eval_tool --help
python -m eval_tool build-dpo --help
```

## 3. 准备输入数据

一个命令可混合读取多个文件。程序按照文件内容自动识别以下容器，不依赖扩展名：

- JSON 数组；
- 单个 JSON 对象；
- 每行一个 JSON 对象的 JSONL。

文件必须使用 UTF-8 编码。

### 3.1 Alpaca 格式

```json
{
  "history": [
    ["<image>\n描述第一张图。", "第一张图的标准答案。"]
  ],
  "instruction": "<image>\n比较第二张图与第一张图。",
  "input": "请只回答关键差异。",
  "output": "当前轮标准答案。",
  "images": [
    "images/first.png",
    "images/second.png"
  ]
}
```

规则：

- `instruction` 和 `output` 必须是非空字符串；
- `input` 可省略、为 `null` 或为空；非空时会以 `instruction + 换行 + input` 组成问题；
- `history` 可省略或为二元素数组组成的列表，每个历史问答也会展开成一个候选样本；
- Alpaca 数据不支持 `system` 字段；
- 单图可使用 `image: "path/to/a.png"`，多图使用 `images: [...]`。

### 3.2 ShareGPT 格式

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\n第一轮问题"},
    {"from": "gpt", "value": "第一轮标准答案"},
    {"from": "human", "value": "<image>\n第二轮问题"},
    {"from": "gpt", "value": "第二轮标准答案"}
  ],
  "images": [
    "images/first.png",
    "images/second.png"
  ]
}
```

规则：

- 对话必须严格按“用户、助手”成对交替；
- 用户角色支持 `human`、`user`，助手角色支持 `gpt`、`assistant`；
- 每个问题和标准答案都必须是非空字符串；
- 每轮都会生成一个候选，当前轮只携带此前的标准答案历史，不会看到未来轮次。

### 3.3 图片规则

- 相对图片路径基于配置中的 `image_root` 解析；
- `<image>` 只统计用户问题中的占位符，数量必须和 `image`/`images` 数量一致；
- 如果提供了图片但问题中完全没有 `<image>`，程序会自动把对应占位符加到第一条用户问题前；
- 正式推理前会读取图片，校验格式、MIME、大小和 SHA-256；
- 最终训练文件保留输入中的原始图片路径，不写入内部解析后的绝对路径。

## 4. 配置文件

先复制示例：

```powershell
Copy-Item dpo.example.json dpo.json
```

完整示例：

```json
{
  "inputs": [
    "data/alpaca_train.json",
    "data/sharegpt_train.jsonl"
  ],
  "output_dir": "outputs/dpo",
  "output_name": "train_dpo.jsonl",
  "work_dir": "work/dpo",
  "image_root": ".",
  "wrong_only": true,
  "rubric": "v4",
  "infer": {
    "model_name": "Qwen2.5-VL-7B-Instruct",
    "model_path": "models/Qwen2.5-VL-7B-Instruct",
    "enable_thinking": false,
    "max_new_tokens": 1024,
    "batch_size": 1,
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "gpu_ids": [],
    "workers_per_gpu": 1
  },
  "judge": {
    "api_base": "http://127.0.0.1:8000/v1/chat/completions",
    "api_key": "REPLACE_WITH_PRIVATE_JUDGE_KEY",
    "model": "local-judge-model",
    "temperature": 0.0,
    "timeout": 120,
    "max_retries": 3,
    "max_workers": 8
  }
}
```

配置采用严格 JSON：字段名写错或出现未知字段会直接报错，不能写注释。

### 4.1 顶层字段

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `inputs` | 是 | 非空输入文件列表。配置内的相对路径以配置文件所在目录为基准。 |
| `output_dir` | 是 | 最终产物目录；不存在时自动创建。 |
| `output_name` | 是 | 训练文件名，必须是安全的 `.jsonl` 文件名，不能包含目录。 |
| `work_dir` | 是 | 推理、Judge 和断点缓存目录；不存在时自动创建。 |
| `image_root` | 否 | 相对图片路径的根目录；省略时使用执行命令时的当前目录。 |
| `wrong_only` | 是 | `true` 只收录 Judge 判错样本；`false` 跳过 Judge。 |
| `rubric` | 条件必填 | `wrong_only=true` 时只能为 `binary` 或 `v4`。 |
| `infer` | 是 | 本地生成模型配置。 |
| `judge` | 条件必填 | `wrong_only=true` 时必须完整提供。 |

除命令行 `--input` 外，配置中的相对路径均以 `dpo.json` 所在目录为基准。命令行 `--input` 的相对路径以当前执行目录为基准。

### 4.2 推理字段

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `model_name` | 无 | 模型身份名称，必填。 |
| `model_path` | 无 | 已存在的本地模型/checkpoint 路径，必填。 |
| `enable_thinking` | `false` | 是否把思考模式传给模型聊天模板。 |
| `max_new_tokens` | `1024` | 单条答案最大生成 token 数，至少为 1。 |
| `batch_size` | `1` | 每批推理样本数，至少为 1；显存不足时优先调小。 |
| `torch_dtype` | `"auto"` | 例如 `auto`、`bfloat16`、`float16`。 |
| `device_map` | `"auto"` | Transformers 模型加载设备映射。 |
| `gpu_ids` | `[]` | 空列表表示单进程；如 `[0, 1]` 则按指定物理 GPU 启动并行 worker。 |
| `workers_per_gpu` | `1` | 每张 GPU 的 worker 数，至少为 1。 |

设置 `gpu_ids` 时，每个 worker 会通过 `CUDA_VISIBLE_DEVICES` 绑定到一张物理卡；`device_map` 应使用 `auto` 或 worker 内的 `cuda:0`，不要写其他 CUDA 编号。

### 4.3 Judge 字段

`wrong_only=true` 时，Judge 配置连接 OpenAI 兼容的 Chat Completions 接口：

| 字段 | 必填/默认值 | 说明 |
| --- | --- | --- |
| `api_base` | 必填 | 完整接口地址，例如 `/v1/chat/completions`。 |
| `api_key` | 必填 | 接口密钥；不要提交真实密钥。 |
| `model` | 必填 | Judge 服务中的模型名。 |
| `temperature` | 必填 | JSON 数值。 |
| `timeout` | 必填 | 单次请求超时秒数，至少为 1。 |
| `max_retries` | 必填 | 失败后的最大重试次数，可为 0。 |
| `max_workers` | `8` | Judge 并发线程数，至少为 1。 |

两种筛选模式：

- `wrong_only=false`：完全跳过 Judge；只要生成答案非空且与 `chosen` 不相同，就可进入训练集。此时 `rubric` 和 `judge` 会被忽略。
- `wrong_only=true`：只有 Judge 明确认定生成答案错误时才进入训练集。接口失败、响应格式错误或无法解析都不会被当作“答错”。

## 5. 推荐运行流程

### 第一步：只做预检

```powershell
python -m eval_tool build-dpo --config dpo.json --dry-run
```

预检会完成 JSON/JSONL 加载、格式归一化、去重、图片读取/MIME/哈希校验，并打印候选数、预推理数、拒绝原因和图片统计。它不会加载模型、不会调用 Judge、不会发布训练文件。

### 第二步：正式构建

```powershell
python -m eval_tool build-dpo --config dpo.json
```

临时覆盖配置中的全部输入文件：

```powershell
python -m eval_tool build-dpo `
  --config dpo.json `
  --input data/alpaca_train.json `
  --input data/sharegpt_train.jsonl
```

只要出现 `--input`，命令行输入就会整体替换配置中的 `inputs`，不是追加。

### 第三步：断点续跑或重建

- 同一份输入和配置中断后，直接重新执行原命令，程序会复用 `work_dir` 中已完成的推理/Judge 缓存；
- 输入、图片、模型/checkpoint 或生成身份发生变化且缓存不匹配时，使用 `--overwrite` 创建新 attempt：

```powershell
python -m eval_tool build-dpo --config dpo.json --overwrite
```

- 希望成功发布后清理可再生中间缓存，可加 `--clean-partial`：

```powershell
python -m eval_tool build-dpo --config dpo.json --clean-partial
```

新结果经过完整校验后才会原子替换旧产物。运行失败或最终选择数为 0 时，已有成功产物不会被覆盖。

## 6. 输出文件

成功后，`output_dir` 中包含：

| 文件 | 说明 |
| --- | --- |
| `train_dpo.jsonl` | 名称由 `output_name` 决定，正式训练数据。 |
| `audit_records.jsonl` | 每条候选的来源、推理、Judge、选择/拒绝状态及原因。 |
| `rejected_records.jsonl` | 所有未进入训练集的审计记录。 |
| `summary.json` | 总数、筛选结果和各类原因统计。 |
| `warnings.log` | 已脱敏的运行警告。 |
| `manifest.json` | 配置摘要、计数以及每个产物的大小和 SHA-256，用于完整性校验。 |

训练文件每行严格只有四个字段：

```json
{
  "conversations": [
    {"from": "human", "value": "问题"}
  ],
  "chosen": {"from": "gpt", "value": "原始标准答案"},
  "rejected": {"from": "gpt", "value": "本地模型生成的错误或较差答案"},
  "images": []
}
```

`chosen` 和 `rejected` 必须是 ShareGPT 助手消息对象，不能写成裸字符串。
在 LLaMAFactory 的 `dataset_info.json` 中按下面方式登记：

```json
{
  "zb_dpo": {
    "file_name": "train_dpo.jsonl",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
      "messages": "conversations",
      "chosen": "chosen",
      "rejected": "rejected",
      "images": "images"
    }
  }
}
```

失败诊断位于：

```text
work_dir/failed_runs/<run_id>/
```

如果失败原因是“没有选出 DPO 样本”，优先查看其中的 `summary.json`、`rejected_records.jsonl` 和 `warnings.log`。

## 7. 常见问题

### 图片占位符数量不匹配

确保所有用户问题中的 `<image>` 总数与 `images` 数量完全一致。若完全不写占位符，程序会自动把全部占位符加到第一条用户问题前。

### 没有样本被发布

常见原因包括输入格式不合规、图片缺失/不可读、重复或参考答案冲突、模型输出为空、模型输出与标准答案完全相同，或者 `wrong_only=true` 时 Judge 未判错。先执行 `--dry-run`，再查看失败诊断目录。

### 显存不足

先把 `infer.batch_size` 调为 1，并按实际 GPU 设置 `torch_dtype`、`device_map` 和 `gpu_ids`。多 worker 会各自加载模型，`workers_per_gpu` 增大也会增加显存占用。

### Transformers 找不到兼容模型类

升级到与本地 Qwen-VL checkpoint 匹配的 `transformers` 版本，并确认模型目录包含完整的配置、权重、processor/tokenizer 文件。

### 推理进度达到 100% 后仍未返回命令行

推理完成后还要依次加载缓存、写入训练与审计文件、校验暂存产物并原子发布。当前版本会输出以下阶段提示：

```text
[DPO] 加载推理缓存...
[DPO] 写入训练文件...
[DPO] 校验暂存产物...
[DPO] 发布最终产物...
[DPO] 构建完成：...
```

这些阶段全部采用流式 JSONL 处理，不会再反复把完整分片读入内存。断点续跑时不要删除 `work_dir/inference`；直接重新执行原命令即可复用已经生成的分片。

`Kwargs passed to processor.__call__ ... processor_kwargs` 是 Transformers 的重复兼容提示。当前版本只屏蔽这一条固定日志，其他 Transformers 警告和异常仍会正常显示。

### Judge 请求失败

确认 `api_base` 是完整的 Chat Completions 地址，服务可访问，`api_key` 和 `model` 正确。必要时增大 `timeout`、`max_retries`，并根据服务容量降低 `max_workers`。

## 8. 安全注意事项

- `dpo.example.json` 中的密钥只是占位符；真实密钥只放在本机配置中；
- 不要把含真实密钥的 `dpo.json`、运行日志或私有数据提交到 Git 或发给无关人员；
- 发布清单会对敏感字段脱敏，但本地原始配置仍需自行保护；
- 第一次处理新数据时始终先运行 `--dry-run`。
