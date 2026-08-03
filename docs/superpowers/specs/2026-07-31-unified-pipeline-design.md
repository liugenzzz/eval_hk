# 统一推理/评估入口设计

日期：2026-07-31  
依据：`统一入口_需求文档.md`。初始开发指令要求第五节 Q1–Q5 按建议默认值实施；2026-08-03 复核要求不把这些选择表述为最终业务确认。当前实现以建议值作为保守默认，其中 Q2 明确采用 fail-safe：推理指纹变化时报错，只有显式 `--overwrite` 才允许全量重跑，绝不自动消耗数小时 GPU。

## 目标与范围

在不改变评分口径的前提下，把数据转换、推理、评估、rubric 切换和多 rubric 对比统一到一份 pipeline 配置与一组子命令中。新入口支持多模型、推理断点续传和严格的预测完整性校验，同时保留现有命令与旧配置格式。

本次包含 R1–R10。`judge.py`、`judge_rubrics.py`、`score_vqa.py`、`aggregate.py`、`report.py`、现有提示词及提示词生成脚本不做与统一入口无关的改动。现有“复用已评分结果”测试只构造 1 行数据，却仍断言通过 2026-07-30 引入的 `MIN_CATEGORY_N = 30` 样本量门槛；这是过期测试 fixture，不是产品回归。实施时把 fixture 补到至少 30 行（或在聚合函数单测中显式传入更小的 `min_category_n`），绝不拆除或绕过统计门槛。

## 方案选择

采用渐进式适配层：新增 `PipelineConfig` 作为新入口的唯一事实源，再派生现有 `InferConfig` 和 `EvalConfig`。相比另建平行 pipeline，它不会复制推理/评估逻辑；相比整体重写，它能保留旧入口、旧配置和现有测试边界。

## 配置模型

pipeline 顶层共享 `tsv_dir`、`datasets`、`work_dir`、`out_dir`、`cache_dir`、`models`、`baseline_model`、`infer` 与 `judge`。模型项包含 `name`、`model_path`，并可用 `pred.<dataset>` 和 `scored.<dataset>` 覆写约定路径。

路径优先级为：

1. `scored.<dataset>`：直接复用已评分明细；
2. `pred.<dataset>`：复用外部预测并跳过推理；
3. `work_dir/<model>/<model>_<dataset_name>.xlsx`：由统一推理入口生成。

所有相对路径均相对于配置文件目录解析。模型过滤在派生配置前完成，并校验未知模型名和重复模型名。若过滤结果不含 baseline，点式评估照常执行，成对评估会被显式关闭并打印原因；不会隐式把 baseline 加回筛选结果。

为补齐需求文档中 `all --config pipeline.json` 未说明转换输入来源的部分，配置允许可选的 `convert.input_json`。`all` 设置该值时先转换；未设置时要求目标 TSV 已存在后再继续。独立 `convert <sharegpt.json> --config pipeline.json` 始终以位置参数为准并写入同一 `tsv_dir`。

## 命令与编排

`eval_tool/cli.py` 提供：

- `convert`：JSON/JSONL 转 TSV；
- `infer`：按模型顺序推理，默认续传；
- `eval`：派生评估配置，可选单一 rubric；
- `sweep`：串行运行多个 rubric，随后生成比较表；
- `all`：可选转换，然后依次推理和评估。

`--models` 对涉及模型的命令进行子集过滤。`--overwrite` 显式从头推理；`--clean-partial` 在成功产出最终文件后清理对应分片。`python -m eval_tool --config <旧配置>`、`python -m eval_tool.run_infer`、`python -m eval_tool.run_eval` 保持原行为；`run_rubric.py` 变为新入口的兼容 shim。

## 推理续传与数据流

每个模型/数据集在任务启动时一次性读取并冻结提示词，随后计算推理指纹：模型路径、冻结后的提示词全文、`max_new_tokens`、dataset key，以及本次实际推理输入（dataset name 与按行序规范化的 index/question/history/image 摘要）的稳定哈希。前四项遵循需求文档给出的公式，输入摘要用于防止 TSV 内容改变但 index 不变时误命中旧预测。分片路径为：

```text
work_dir/<model>/_partial/<dataset>__<infer_fp>[__w<worker>].jsonl
```

记录格式为 `{"index": "...", "prediction": "..."}`。数据载入时先拒绝空或重复 index。顺序模式逐批校验生成器返回数等于 batch 行数，再立即追加并 flush；并行模式由每个 worker 写自己的分片。启动时扫描所有 worker 分片，以字符串化 dataset index 去重，跳过已完成行，并打印完成数；改变 worker 数量或 GPU 拓扑不影响重载。模型真正返回的空字符串属于“已返回的空答案”，不会与“缺失记录”混为一谈。

若发现同一模型/数据集存在其他指纹的保留分片或无法证明现有最终 xlsx 属于当前指纹，默认报错并提示 `--overwrite`，避免静默执行耗时的全量重跑或复用旧结果。显式覆盖只重置当前目标运行状态；历史分片默认保留供核查。

最终 xlsx 先写同目录临时文件，成功关闭后原子替换目标；随后原子写入同目录的 `<model>_<dataset>.infer.json` 清单，记录指纹、数据集名称、总行数和完成的 index 摘要。这样即使用户用 `--clean-partial` 清掉 JSONL，后续仍能验证最终文件是否匹配当前推理配置。约定路径中已存在但没有清单的旧 xlsx 不会被新 pipeline 静默信任：用户可将其显式声明为 `pred`，或用 `--overwrite` 重新生成。

完成后按真值行序合并。缺失、重复、越界或未知 index 都写入模型输出目录的 `warnings.log` 并抛错，不生成或覆盖最终 xlsx。worker 异常时继续收集其他 worker 的完成状态，已落盘 batch 可供下次续传，最后汇总异常并失败退出。

## Rubric 管理与比较

rubric 注册表把 `v1/v2/v3/v3b/v4/v4b` 映射到现有 pointwise 提示词。派生评估配置只替换 pointwise prompt 与带 rubric 后缀的输出目录，保留 judge fingerprint 输出和其余评估参数。

`sweep` 严格串行调用裁判服务。因为 `scored` 文件没有可验证的 rubric 元数据，sweep 对所选模型发现任何 `scored` 覆写都会报错，避免把同一份旧裁判结果伪装成多组 rubric。各版本产出 `<out_dir>_<rubric>/judge_detail_all.xlsx` 后，调用从 `compare_rubrics.py` 抽出的纯函数，对每个非 baseline 模型分别生成并打印比较 DataFrame，并把结果写到共同父目录中的 `rubric_comparison.csv`。任一 rubric 失败时停止后续比较，保留已完成报告与裁判缓存。

## 错误处理与兼容性

- 配置加载对缺失必填字段、未知数据集/模型/rubric、重复名称给出带字段路径的错误；
- 外部 `pred` 与 `scored` 路径继续优先，且不要求对应模型执行推理；
- 旧 loader 和 dataclass 的现有默认值不变，新 pipeline 使用“默认续传”的新默认值；
- 多模型在同一组 GPU 上串行执行，任一模型失败即停止当前命令；此前已完成的模型产物和失败模型的分片都保留；
- 所有危险状态先失败，绝不以空字符串补齐预测；
- 推理 warnings 固定写到 `work_dir/<model>/warnings.log`，评估 warnings 继续写到其 rubric 输出目录；均使用 UTF-8 追加写，异常信息同时打印到终端；
- 同一模型/数据集/指纹的并发推理通过独占锁拒绝第二个进程，避免两个进程同时写 `w0` 分片。

## 测试与验收

按测试驱动顺序增加：

1. pipeline 配置解析、相对路径、派生配置、覆写优先级与模型过滤；
2. 指纹稳定性、JSONL 重载、按 batch 持久化、默认续传、显式覆盖和分片清理；
3. 严格合并在缺失/重复 index 时失败，worker 部分成功可恢复；
4. 多模型依次推理并把约定预测路径自动交给评估；
5. rubric 内存派生、串行 sweep、比较函数与落盘结果；
6. 新 CLI、`all` 编排以及全部旧入口/旧配置回归；
7. README、开放问答使用说明和 pipeline 示例中的命令冒烟测试。

验收以无真实 GPU/裁判 API 的 fake generator/client 单元与集成测试为主；最终运行全量 pytest 和现有 `e2e_check.py` 的离线通路。真实模型或裁判服务不可用时明确报告未执行项，不把离线替身结果表述为线上验证。
