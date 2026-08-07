# 直接 JSON/JSONL DPO 构建功能设计

日期：2026-08-07
状态：方案已确认，书面规格待用户复核，尚未进入实现
参考脚本：`F:/AI-Haishi/智航院/script/build_dpo.py`、`F:/AI-Haishi/智航院/script/build_dpo_binary.py`

## 1. 背景

现有两个参考脚本都是“已评分明细表到 DPO 数据”的后处理器：

- `build_dpo.py` 读取 v4 五维度评分结果；
- `build_dpo_binary.py` 读取 binary 正确/错误评分结果。

它们不负责原始数据读取后的模型推理和正确性评估，而且当前行为固定为只取错题、要求样本带图，并依赖外部流程先产生 XLSX 或 JSONL 明细。新功能要把两套后处理能力整合为一个独立入口，直接完成：

```text
多个 JSON/JSONL
→ 格式识别与逐轮规范化
→ 本地模型推理
→ 可选 Judge 评估
→ 错题或全量筛选
→ 单个 ShareGPT DPO JSONL
```

整个过程不得先转换为 TSV，也不得把 XLSX 作为中间协议。

## 2. 目标

新增独立命令：

```powershell
python -m eval_tool build-dpo --config dpo.json
```

该命令应满足：

1. 一次读取多个 `.json` 或 `.jsonl` 文件；
2. 同一任务可混合 Alpaca 与 ShareGPT 数据；
3. 支持纯文本、单图、多图和多轮对话；
4. 直接使用本地 HuggingFace/Qwen-VL 模型目录推理；
5. Judge 支持 `binary` 与 `v4` 两种二选一的 rubric；
6. 支持“只构建错题”和“跳过评估、构建全部”两种模式；
7. 最终只产生一个训练数据 JSONL，统一为 ShareGPT DPO 格式；
8. 推理与 Judge 均可断点续跑；
9. 图片路径字符串保持不变，不复制图片，不改写路径；
10. 未入选样本及其原因可审计，不静默丢弃。

`rejected` 必须由本次配置的本地模型实际生成，不接受外部文件伪装为本次预测；manifest 记录其模型/checkpoint 身份。DPO 训练应从同一模型身份出发，是否选择同一 checkpoint 由训练任务负责人负责。

## 3. 非目标与范围边界

本功能不包含：

- JSON/JSONL 到 TSV 的转换；
- XLSX 预测或评分明细的生成与消费；
- 多个被推理模型的一次性横向比较；
- binary 与 v4 在同一任务中同时评估同一条样本；
- 类别均衡采样、每类数量上限或固定总量抽样；
- 从错题中额外划分验证集；
- 图片复制、base64 回落文件生成或图片路径重写；
- 图形化文件选择器；
- 自定义第三种 rubric；
- 自动判断参考答案本身是否高质量。

`wrong_only=false` 会有意把所有有效模型回答写成 `rejected`，其中可能包含与参考答案语义等价的正确回答。这是用户确认的全量构建语义；程序只排除字符串完全相同的偏好对，不尝试用 Judge 或语义相似度再次过滤。

## 4. 已确认的产品口径

### 4.1 输入文件

- 配置通过 `inputs` 数组选择多个文件；
- 命令行可重复传入 `--input`，一旦出现便整体替换配置中的 `inputs`；配置内输入路径相对配置文件目录解析，命令行输入路径相对当前工作目录解析；
- 文件按配置或命令行给定的顺序处理；
- 文件内记录顺序和多轮回合顺序保持稳定；
- 相对配置路径相对于配置文件所在目录解析；
- 图片路径不参与上述路径解析，读取、校验和输出均使用数据中原始字符串。

内容识别按文件内容而非只按扩展名进行：

1. 顶层 JSON 数组；
2. 顶层单个 JSON 对象；
3. 每个非空行一个 JSON 对象的 JSONL。

格式化后跨多行的 JSON 数组属于第 1 类。跨多行的单条 JSONL 记录不属于合法 JSONL，不予兼容。

### 4.2 Alpaca 适配

Alpaca 基本字段为：

```json
{
  "instruction": "任务说明",
  "input": "可选输入",
  "output": "参考答案",
  "history": [["可选历史问题", "可选历史答案"]],
  "images": ["可选图片路径"]
}
```

规范化规则：

- `instruction` 必填；
- `input` 为空时只使用 `instruction`；
- `input` 非空时，按 `instruction + "\n" + input` 组成当前 human 内容；
- `output` 作为 `chosen`；
- 可选 `history` 必须是问题/答案二元组列表；存在时将这些历史回合与当前 `instruction/input/output` 组成完整对话，并按“每轮一条”的同一规则展开；
- 本期不支持 Alpaca 顶层 `system`；出现时整条记录排除并审计，不能静默丢弃系统指令；
- 单图字段可写成 `"image": "path"`，多图或统一形式使用 `"images": ["path"]`；
- 同时出现 `image` 和 `images` 时，规范化后完全相同才接受并只保留一份；内容不同则整条记录按图片字段冲突排除；
- `image`/`images` 均可省略，省略时按纯文本样本处理；
- 没有 `history` 的 Alpaca 记录产生一个 DPO 候选样本；有 `history` 时每个历史回合和当前回合分别产生候选样本；
- 图片仍按各轮 human 文本中的 `<image>` 顺序关联，后续回合图片不能进入前序样本。

### 4.3 ShareGPT 适配

ShareGPT 基本字段为：

```json
{
  "id": "可选来源 ID",
  "conversations": [
    {"from": "human", "value": "<image>\n问题"},
    {"from": "gpt", "value": "参考答案"}
  ],
  "images": ["原始图片路径"]
}
```

规范化规则：

- 支持 `human/gpt`，并兼容等价的 `user/assistant` 角色名；
- 单图字段可写成 `image`，多图或统一形式使用 `images`，内部一律规范化为路径数组；
- 同时出现 `image` 和 `images` 时采用与 Alpaca 相同的“一致才接受、冲突即排除”规则；
- 本期只接受上述 user/assistant 回合角色；出现 `system`、`tool` 或其他角色时整条原始记录排除并审计，不静默忽略；
- 对话必须能组成 human→gpt 回合；
- 任一位置角色错序、问题/答案缺失或历史损坏时整条原始记录排除，不允许跳过异常消息后把后续消息重新配对；
- 每个回合独立产生一条 DPO 候选样本；
- 当前回合之前的标准问答作为历史上下文；
- 当前 human 是 `conversations` 的最后一条；
- 当前原始 gpt 回答从上下文中移出并作为 `chosen`；
- 被推理模型只生成当前回合回答，作为 `rejected`；
- 历史始终使用原始标准答案，不使用模型先前生成的回答；
- `user/assistant` 输入角色在最终训练数据中统一写为 `human/gpt`；
- 因此各回合可独立推理、并行、缓存，且前一轮预测错误不会污染后一轮。

多轮多图按 `<image>` 在对话前缀中的出现顺序关联 `images`。为某一回合构建样本时，只携带截至当前 human 已引用的图片，不能把后续回合图片提前暴露。路径字符串保持原样。

### 4.4 图片规则

- `images` 是可选字段；
- 无图样本输出 `"images": []`；
- 有图样本必须能通过下述读取基准找到文件，但输出仍保留原始路径字符串；
- 有 `<image>` 占位符时，其总数必须与记录的图片总数一致，并按出现顺序绑定；
- 图片非空但整条原始对话没有 `<image>` 时，按兼容模式把全部图片视为从第一轮开始共享的全局图像，并在规范化后的第一条 human 消息前补齐对应数量的 `<image>`；
- 已存在至少一个占位符但数量不匹配时，不能猜测图片归属，该记录不进入训练集；
- 推理时可在内存中打开图片；Judge 请求可在内存中临时编码图片；两者均不得改写训练数据里的路径。

图片读取基准由可选顶层配置 `image_root` 指定；省略时使用命令运行目录。绝对图片路径直接读取，相对图片路径仅相对该基准解析，不尝试逐个输入文件目录。`image_root` 只影响读取和校验，绝不改变输出中的路径字符串。

送入本地模型或 Judge 时，适配层把 `<image>` 与对应图像转换为一次且仅一次的多模态内容；不能既保留一个会被处理器再次解释的字面占位符，又额外挂载同一张图片。每张图片必须根据真实文件内容得到独立 MIME，不能把 PNG/JPEG 混合输入统一伪装成 `image/jpeg`；Judge 的结构化消息必须按占位符位置交错插入图片，不能把所有图片统一追加到文本末尾。

## 5. 输出格式

所有有效输入统一输出为一个 ShareGPT DPO JSONL。每行严格使用以下训练结构：

```json
{
  "conversations": [
    {"from": "human", "value": "问题或历史"}
  ],
  "chosen": {"from": "gpt", "value": "参考答案"},
  "rejected": {"from": "gpt", "value": "被推理模型回答"},
  "images": []
}
```

有图时 `images` 为原始路径数组。来源文件、来源 ID、回合号、评分和失败原因不得写入训练 JSONL，而应写入审计产物，避免训练工具受到额外字段影响。

推理直接使用规范化后的原始会话，不套用现有 TSV 数据集的 `infer_vqa*.txt` 包装提示词，也不额外注入通用任务说明。输入文件若有业务指令，应由其 `instruction`、`input` 或 `conversations` 自身提供。

所有 JSONL 产物使用 UTF-8、`ensure_ascii=False` 和一行一个完整 JSON 对象。

## 6. 评估与筛选语义

### 6.1 全量模式

```json
{"wrong_only": false}
```

- 对全部有效候选执行本地模型推理；
- 不创建 Judge 客户端，不发起 Judge 请求；
- `rubric` 与 `judge` 配置均可省略；
- 通过基础质量过滤的候选全部进入最终 DPO JSONL。

### 6.2 错题模式

```json
{
  "wrong_only": true,
  "rubric": "binary"
}
```

或：

```json
{
  "wrong_only": true,
  "rubric": "v4"
}
```

- 全部有效候选先执行推理；
- 再调用 Judge；
- `binary` 只保留 `correct=false` 的样本；
- `v4` 依据未舍入质量分计算的 `hit` 筛选，只保留 `hit=0`；未舍入分数恰好 60 视为通过；
- 两位小数的 `quality_score` 只用于展示，不能反向决定入选；允许审计中出现显示为 `60.0` 但因未舍入值略低于 60 而 `hit=0` 的边界记录；
- Judge 调用或响应解析失败不是错题，不进入训练集；
- 保留全部评估不通过样本，不按错误类型限量或均衡采样。

多轮样本的 Judge 输入必须包含此前全部标准问答、当前 human、当前 `chosen`、当前 `rejected`，以及截至当前轮可见的图片。只把“当前问题”孤立发送给 Judge 会丢失上下文，禁止这样实现。完整 Judge 输入摘要属于缓存键。

### 6.3 图文提示词分流

rubric 与样本模态是两个独立维度：

| rubric | 有图样本 | 无图样本 |
|---|---|---|
| `binary` | 多模态正确/错误提示词 | 纯文本正确/错误提示词 |
| `v4` | 现有装备多模态 v4 语义 | 纯文本 v4 语义 |

`binary` 是配置层名称，Judge 输出继续使用现有兼容 schema：`correct` 与 `reason`（可带现有解析器认识的 binary/v1 标记），并复用现有二值解析能力。纯文本 binary 只比较当前任务、参考答案和模型回答，不包含“必须看图”的判据。

纯文本 v4 保持与现有解析器兼容的输出字段，并把字段语义扩展到通用文本任务：

- `fact_score` 定义为“核心任务完成度与语义正确性”，所有纯文本任务都必须返回 0–100 数字；即使是改写、创作或格式转换任务也不能为 `null`；
- `param_score` 只在任务涉及数值或明确属性参数时评分，否则为 `null`；
- `fabrication_score`、`style_score` 正常评分；
- `visual_score` 必须为 `null`，不参与加权，剩余准确性维度重新归一；
- 为兼容现有 v4 schema，`equipment_correct` 在文本模板中定义为“是否正确理解任务对象或主体”，不能要求 Judge 假装看图；
- 质量分继续复用现有 v4 计算和 60 分通过阈值。

binary 与 v4 每次任务只能选择一个，不同时运行。

在调用现有 parser 前，DPO 层必须做严格 Judge schema 校验：binary 的 `correct`、v4 的 `equipment_correct` 只接受 JSON 布尔值；各分数字段只接受规定范围内的 JSON 数字或明确允许的 `null`。`null`、任意字符串或其他类型不能通过“不是 true 就当 false”的宽松转换进入错题集，而应记为 `judge_error`。

## 7. 基础质量过滤

以下规则独立于 `wrong_only`，在两种模式中始终生效：

1. 缺少问题或 `chosen`：排除；
2. 对话角色不成对或回合无法解析：排除；
3. 引用了图片但文件不存在：排除；
4. `<image>` 数量与图片数量不一致：排除；
5. 本地模型推理失败或返回空字符串：排除；
6. `chosen` 与 `rejected` 去除首尾空白后完全相同：排除；
7. 错题模式下 Judge 最终失败：排除；
8. 错题模式下评估通过：排除。

不沿用旧脚本的以下硬限制：

- chosen/rejected 至少 8 个字符；
- chosen/rejected 字符长度比不超过 2.2。

原因是通用 Alpaca 数据可能存在合法短答案，字符长度比也不等价于偏好质量。

## 8. 去重与冲突

规范化后，为每个候选计算稳定摘要。

- 完全重复键：`conversations + chosen + images`；
- 冲突检测键：`conversations + images`。

处理规则：

- 完全重复样本只保留按输入顺序出现的第一条；
- 同一冲突检测键出现多个不同 `chosen` 时，这些记录全部不进入训练集；
- 重复和冲突均记录全部来源位置；
- 去重发生在模型推理前，避免浪费 GPU；
- 不做语义去重或近似文本去重。

审计用 `sample_id` 由输入内容摘要、记录序号、回合序号和规范化内容共同派生；不得只依赖可能重复或缺失的原始 `id`。

## 9. 配置设计

示例配置：

```json
{
  "inputs": [
    "data/a.jsonl",
    "data/b.json"
  ],
  "output_dir": "dpo_output",
  "output_name": "merged_dpo.jsonl",
  "work_dir": "dpo_work",
  "image_root": ".",
  "wrong_only": true,
  "rubric": "v4",
  "infer": {
    "model_name": "base",
    "model_path": "models/Qwen3-VL",
    "enable_thinking": false,
    "max_new_tokens": 1024,
    "batch_size": 1,
    "torch_dtype": "auto",
    "device_map": "auto",
    "gpu_ids": [],
    "workers_per_gpu": 1
  },
  "judge": {
    "api_base": "http://127.0.0.1:18180/v1/chat/completions",
    "api_key": "sk-local",
    "model": "judge-model",
    "temperature": 0.0,
    "timeout": 120,
    "max_retries": 3,
    "max_workers": 8
  }
}
```

校验规则：

- `inputs` 非空；
- `output_name` 必须以 `.jsonl` 结尾；
- `image_root` 可省略；配置中提供时相对配置文件目录解析，省略时固定为命令运行目录；
- `wrong_only` 必填且必须是 JSON 布尔值，不接受字符串、数字或隐式默认；
- `infer.model_name` 与 `infer.model_path` 必填；
- `infer.model_path` 必须存在；
- `infer.enable_thinking` 为可选严格布尔值，默认 `false`；它必须传给实际 chat template，并写入指纹与 manifest，避免 `rejected` 无意带 `<think>` 而 `chosen` 不带时产生格式偏好；
- `wrong_only=true` 时，`rubric` 必须为 `binary` 或 `v4`，且 `judge` 必填；
- `wrong_only=false` 时忽略已有 `rubric`/`judge`，并明确打印“Judge 已跳过”；
- 所有相对配置路径相对于配置文件目录解析，图片路径除外；
- 日志和 manifest 不得写出 `api_key` 明文。

### 9.1 命令行

```powershell
python -m eval_tool build-dpo --config dpo.json
python -m eval_tool build-dpo --config dpo.json --input a.jsonl --input b.json
python -m eval_tool build-dpo --config dpo.json --dry-run
python -m eval_tool build-dpo --config dpo.json --overwrite
python -m eval_tool build-dpo --config dpo.json --clean-partial
```

- `--input` 可重复，出现时整体替换配置输入列表；
- `--dry-run` 只加载、规范化、拆轮、校验格式与图片、统计去重和冲突，不加载本地模型，不调用 Judge；
- `--overwrite` 重置当前任务的推理与 Judge 续跑状态；
- `--clean-partial` 只在最终输出成功后清理可再生的中间缓存。

## 10. 模块边界

新增模块建议：

```text
eval_tool/
  dpo_config.py     # 专用配置 dataclass、加载与校验
  dpo_input.py      # JSON/JSONL 识别、Alpaca/ShareGPT 适配、逐轮规范化
  dpo_prompts.py    # binary/v4 × 图/文提示词路由
  dpo_pipeline.py   # 去重、推理、评估、筛选、续跑、导出编排
  dpo_report.py     # audit/rejected/summary/manifest 产物
```

现有模块复用边界：

- 复用 `eval_tool.infer` 的模型加载、生成参数与并行运行思路；
- 新增直接会话和图片路径适配层，不复用 TSV row 或 XLSX 输出协议；
- 复用 `eval_tool.judge.JudgeClient` 的 OpenAI 兼容传输和响应解析思路，但新增结构化多轮多图请求方法，负责逐图 MIME 和占位符交错；不能直接复用当前“单 MIME、图片全部追加到文本末尾”的 content 组装；
- 复用 `eval_tool.judge_rubrics` 的 binary/v4 解析、v4 质量分和阈值；
- 复用现有 JSONL 分片、指纹和原子发布思路，但为 DPO 新增严格缓存实现；不能直接复用会跳过坏行或允许多个线程无协调追加同一文件的宽松缓存行为；
- 不修改现有 TSV `infer`、`eval`、`all` 命令的输入输出语义。

## 11. 断点续跑与指纹

默认启用断点续跑，推理缓存与 Judge 缓存分离。

推理指纹至少包含：

- 按顺序排列的输入内容摘要；
- 规范化和逐轮拆分规则版本；
- `model_name`、规范化后的 `model_path`；
- checkpoint 身份：模型配置文件内容摘要，以及权重文件相对名、大小和高精度修改时间的清单摘要，用于识别同一路径下被替换的权重；
- `enable_thinking`、`max_new_tokens`、batch/GPU/精度等影响生成的配置；
- 实际送入模型的上下文、图片路径摘要和图片文件内容摘要。

Judge 使用两层指纹，不能合成一个：

`judge_request_fp` 至少包含：

- 对应推理指纹；
- Judge `api_base`（不包含密钥）；
- Judge 模型名；
- rubric；
- 图文提示词全文摘要；
- temperature 和影响请求的参数；
- 完整 Judge 文本、多轮消息、逐图 MIME 和图片内容摘要。

`judge_parse_fp` 包含 `judge_request_fp`，并额外包含 Judge 严格 schema、解析器和 v4 计分规则版本。

行为：

- 输入、模型或生成参数变化时，旧推理缓存不得静默复用；
- 发现不匹配状态时默认失败并提示 `--overwrite`，不自动消耗 GPU 全量重跑；
- 只修改 rubric、Judge 模型或提示词时，复用推理缓存，仅重新评估；
- 每条完成的推理与 Judge 结果立即追加并 flush；
- Judge 原始响应按 `judge_request_fp` 缓存，解析结果按 `judge_parse_fp` 缓存；解析/计分规则变化时可在 schema 兼容的前提下重新解析原始响应，只有请求身份变化才重新请求 API；
- 每个缓存文件只能由一个写入协调器负责，或按 worker 分片后再严格归并；读取时只能修复可确认的截断尾行，文件中部坏 JSON、重复冲突或未知 sample ID 必须报错，不能静默跳过；
- append 后立即 flush，并按有界批次执行 `fsync`，命令正常结束和原子发布前必须 `fsync`；
- 最终输出按规范化候选原始顺序合并，不按任务完成顺序输出；
- 所有交付产物先写入本次运行专属 staging 目录并校验；训练 JSONL 记录 SHA-256、字节数和行数，`manifest.json` 最后发布，作为这组产物的提交标记；
- 单文件均采用同目录临时文件加原子替换；若多文件发布中断，新旧 manifest 与训练文件摘要不匹配时下次启动必须拒绝静默复用；
- 旧的已完成训练文件没有匹配 manifest 时不得被静默信任或覆盖。

## 12. 错误处理

### 12.1 启动前终止

以下错误在加载大模型前终止：

- 配置无法解析或必填字段缺失；
- 输入文件不存在、不可读或整个文件不是合法 JSON/JSONL；
- 输出目录或 work_dir 不可创建；
- 本地模型路径不存在；
- 错题模式缺少合法 Judge 配置。

### 12.2 单条排除并继续

以下错误不拖垮其他记录：

- 单条记录 schema 不合法；
- 问题、参考答案或回合缺失；
- 图片缺失或占位符不匹配；
- 完全重复或参考答案冲突；
- 单条推理失败、空回答或相同回答；
- Judge 达到最大重试次数仍失败；
- Judge 返回无法解析或缺字段的 JSON；
- 错题模式下样本评估通过。

Judge 请求使用配置的最大重试次数并退避。Judge 错误不能被当成 `correct=false` 或 v4 低分。

本地批量推理异常时，若不是 OOM、CUDA 上下文损坏、模型进程退出等基础设施错误，先把失败 batch 递归二分或回退为单条推理；只有收敛到单条仍失败时，才把该样本记为 `inference_error` 并继续。OOM、GPU/worker 退出和模型不可用属于运行级错误，不能伪装成若干单条坏数据继续运行。

### 12.3 运行级终止

模型加载失败、OOM/CUDA 上下文损坏、GPU worker 崩溃、缓存或最终输出无法写入时终止命令，已成功 flush 的中间结果保留供续跑。不得用空回答填补缺失记录。

若最终有效样本数为零，命令返回失败，不创建或覆盖训练 JSONL、manifest、audit、rejected、summary 等既有成功交付产物；但必须在 `work_dir/failed_runs/<run_id>/` 写出本次独立的 `audit_records.jsonl`、`rejected_records.jsonl`、`summary.json` 和错误摘要，并在终端打印路径，保证全量排除仍可诊断。

## 13. 输出产物

交付目录：

```text
dpo_output/
  merged_dpo.jsonl
  audit_records.jsonl
  rejected_records.jsonl
  summary.json
  manifest.json
  warnings.log
```

- `merged_dpo.jsonl`：严格训练数据，一行一个 DPO 样本；
- `audit_records.jsonl`：每个规范化候选的来源、回合、摘要、prediction、Judge 结果与最终 disposition；无法规范化但语法上合法的原始记录也必须写入来源文件、JSONL 行号或数组下标、原始内容摘要和失败原因；
- `rejected_records.jsonl`：所有未入选项及机器可读原因码；语法合法但无法规范化的原始记录也必须写入，不以其尚未成为“候选”为由遗漏；
- `summary.json`：按输入文件、原格式、纯文本/单图/多图、单轮/多轮、通过/排除原因聚合计数；
- `manifest.json`：脱敏配置、输入摘要、推理/Judge 指纹，以及最终训练 JSONL 的 SHA-256、字节数和行数；
- `warnings.log`：可读告警和运行异常摘要。

中间推理/Judge JSONL 缓存存放在 `work_dir`，不混入交付目录。`--clean-partial` 不删除 manifest、summary、audit 或 rejected 产物。

核心排除原因码固定为：`invalid_schema`、`unsupported_role`、`conflicting_image_fields`、`missing_question`、`missing_chosen`、`unpaired_turns`、`missing_image`、`image_placeholder_mismatch`、`duplicate`、`reference_conflict`、`inference_error`、`empty_rejected`、`identical_pair`、`judge_error`、`judge_pass`。可以增加更细的子原因，但 summary 必须能按这些稳定主原因聚合。重复副本以及参考答案冲突中的每个来源都必须分别留下一条 disposition，不能只记录聚合数量。

## 14. 测试设计

### 14.1 输入与规范化单元测试

- 顶层数组、单对象、JSONL、格式化多行 JSON；
- 无效整文件与单条坏记录的不同处理；
- Alpaca 的空/非空 `input` 与可选 `history` 逐轮展开；
- ShareGPT 的 `human/gpt` 与 `user/assistant`；
- 不支持的 `system/tool` 角色进入审计而不被静默忽略；
- 历史损坏时整条记录排除，不能跳过坏消息后重新配对；
- `image` 与 `images` 一致时合并、冲突时排除；
- 纯文本、单图、多图；
- 多轮逐轮拆分和标准答案历史；
- 多轮图片不得提前泄露；
- 图片路径字符串保持不变；
- 占位符与图片数量校验；
- 有图片但零占位符时补到第一轮，已有占位符但数量不符时排除；
- `image_root` 只影响读取，输出路径字符串不变；
- 稳定顺序、精确去重和答案冲突。

### 14.2 推理与评估单元测试

使用 fake generator 和 fake Judge，不依赖真实 GPU/API：

- 当前回合模型输入包含正确历史；
- `enable_thinking=false` 默认实际传入 chat template，开启/关闭均进入指纹；
- batch 的数据级失败可二分回退到单条，OOM/worker 退出仍为运行级失败；
- 多轮 Judge 输入包含完整标准答案历史和截至当前轮的图片；
- prediction 正确落到 `rejected`；
- `wrong_only=false` 不实例化、不调用 Judge；
- binary 有图/无图提示词分流；
- v4 有图/无图提示词分流；
- 文本 v4 的 `visual_score=null`；
- 纯文本改写/格式任务的 `fact_score` 必须有任务完成度数值，不会因所有准确性维度为 null 而解析失败；
- binary 只收 `correct=false`；
- v4 按未舍入值产生的 `hit` 筛选；覆盖显示 `quality_score=60.0` 但 `hit=0` 的舍入边界；
- Judge 错误不作为错题；
- `correct`/`equipment_correct` 的 null、字符串或错误类型被严格拒绝，不能宽松转换为 false；
- PNG/JPEG 混合多图使用各自 MIME，并按多轮占位符位置交错进入 Judge 请求；
- 空回答和相同回答排除。

### 14.3 续跑与产物测试

- 每条结果追加并可从中断状态恢复；
- 改 GPU worker 数不破坏结果归并；
- 输入/模型/生成参数变化拒绝旧推理缓存；
- 同一路径下图片内容或 checkpoint 文件身份变化拒绝旧推理缓存；
- 只改 rubric 或 Judge 配置时复用推理；
- Judge request/parse 双层指纹允许只重解析原始响应；
- 缓存中部坏行或并发冲突会失败，只有可确认的截断尾行可修复；
- 最终输出顺序与任务完成顺序无关；
- 原子写入失败不破坏上一份完整结果；
- API key 在日志和 manifest 中脱敏；
- manifest 的输出 SHA-256、字节数和行数与训练文件一致，并作为多产物发布提交标记；
- audit、rejected、summary 计数一致；
- 无法规范化的合法 JSON 记录同时进入 audit 和 rejected；每个重复来源和每个冲突来源均有独立 disposition；
- 零有效样本不覆盖旧训练文件。
- 零有效样本在独立 failed_runs 目录保留 audit、rejected 和 summary，并返回其路径。

### 14.4 CLI 与回归测试

- `build-dpo --config`；
- 重复 `--input` 的整体覆盖行为；
- `--dry-run`、`--overwrite`、`--clean-partial`；
- 错题模式配置校验；
- `wrong_only` 缺失或不是 JSON 布尔值时启动前失败；
- 全量模式省略 Judge；
- 现有 `convert`、`infer`、`eval`、`sweep`、`all` 行为不变；
- 运行现有完整 pytest 与离线端到端检查。

真实本地模型和 Judge API 只做小规模可选冒烟测试。没有实际运行线上测试时，交付报告必须明确写“未执行”，不能用 fake 测试冒充真实服务验证。

## 15. 验收标准

1. 一条命令读取多个混合 JSON/JSONL，全程不创建 TSV 或 XLSX；
2. 最终生成一个可训练的 ShareGPT DPO JSONL；
3. Alpaca、ShareGPT、纯文本、单图、多图和多轮均正确处理；
4. ShareGPT 每轮独立构建，使用标准答案历史；
5. 图片路径内容保持原样，后续图片不泄露到前序回合；
6. `wrong_only=true` 支持 binary/v4，且保留全部错题、不限类别数量；
7. `wrong_only=false` 不调用 Judge；
8. 推理和 Judge 中断后只补未完成项；
9. 改 Judge/rubric 不重跑推理，改输入/模型/生成参数不误用旧推理；
10. 无效偏好、重复、冲突、图片缺失和 Judge 错误不会进入训练集；
11. 每条排除记录均能从审计产物追溯原因；
12. 现有 TSV 推理与评估流程无回归。

## 16. 实施前约束

- 本文档获用户复核批准后，才能编写实施计划；
- 实施计划获准前不修改功能代码；
- 实现时按测试驱动顺序先写失败测试，再写最小实现；
- 不顺带重构与本功能无关的旧评估逻辑；
- 只复用稳定公共能力，不把新的 JSON/JSONL 直读流程重新耦合回 TSV。
