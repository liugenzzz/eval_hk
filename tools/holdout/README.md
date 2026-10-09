# tools/holdout：评估留出、训练导出、泄漏检查

评估集里的图、视频、书一律不得进训练。field 库靠两样东西实现这条规则：

- 登记表 `field.eval_holdout`：哪个评估集留出了哪张图、哪段视频、哪本书或哪一期；
- 训练表的 `is_holdout=1` 标记：被留出的行打上标记，不删除。`corpus_samples` 和 `defect_tetect_samples` 都有这一列，对应的视图 `v_train_samples` / `v_train_defect_tetect_samples` 会排除这些行。

**标记只有在导出时被过滤才起作用。** 2026-09-30 那次导出没有过滤，32B 训练集因此混进了书籍、期刊、园区、飞机评估集。这里的脚本把「登记 → 导出 → 检查」串成一条固定流程。

| 脚本 | 做什么 | 改不改库 |
|---|---|---|
| `register_eval_sets.py` | 三种登记方式。① 默认：按 md5 登记工业 `industrial_eval_v1` 和检测 `mbjc_test_v1`，首次运行会给 `defect_tetect_samples` 加 `is_holdout` 列并建视图。② `--copies <tsv>`：登记跨池副本。③ `--books <评估集> <批次> <清单>`：按书名或期名整本、整期留出。三种方式都会核对已登记评估集的标记有没有漏 | 默认只审计；`--apply` 才写 |
| `find_copies.py` | 拿每个评估集的图和视频，去和库里**所有**媒体池比对：先找同文件名、同大小的候选，再用 md5 确认，输出副本清单 | 只读 |
| `build_journal_eval.py` | 从一个期刊批次抽评估集：整期留出，按年代分层抽期，每类任务最多 N 条，同时输出期名清单 | 只读 |
| `export_train_data.py` | 导出训练文件，按「语料类型 / pt·qa·vqa」两级分目录。跳过 `is_holdout=1` 的行，以及 `--exclude-media` 里列出的媒体（兜底用）；`--bench` 先估时间。只有加 `--dataset-info` 才生成 LlamaFactory 注册文件 | 只读 |
| `splice_export.py` | 数据或剔除规则变了之后，只重导受影响的批次，再拼回原导出；合并过的 pt 文件按 id 前缀替换 | 只读 |
| `rename_sources.py` | 按 `SOURCE_NAMES` 给已有导出的文件改成好认的名字，同步改 manifest | 只读 |
| `summarize_export.py` | 生成导出目录的 `README.md`：说明是全量数据、使用和更新规范、统计、对账、泄漏检查、更新记录 | 只读 |
| `eval_copies.py` | 从 md5 缓存里列出和评估图内容相同的训练图（早期做法，已被 `find_copies.py` 取代） | 只读 |
| `verify_export.py` | 校验导出结果：格式、占位符、行数对账、媒体抽查、`dataset_info` 一致性 | 只读 |
| `leak_check.py` | 拿全部评估集和导出里的 qa/vqa 文件逐行比对：答案原文、媒体路径和 md5、库 id。**每次导出后、训练前都要跑一遍** | 只读 |
| `fielddb.py` / `mdl.py` | 数据库连接配置（从 `$FIELD_DB_CONF` 读，默认 `mm_env/field_tools/db.json`，不进仓库）/ 在共享表上安全执行 DDL | — |

## 导出产物结构

```
<out>/<语料类型>/pt/<语料类型>.pt.jsonl     # {"id", "text"}，同一语料类型的 pt 全部合成一个文件
<out>/<语料类型>/qa/<来源>.<格式>.jsonl     # 纯文本问答；格式为 sharegpt | openai | alpaca
<out>/<语料类型>/vqa/<来源>.<格式>.jsonl    # 带图或带视频的问答
<out>/README.md                             # summarize_export.py 生成
<out>/manifest.json                         # 每个文件：行数、字节、格式、模态、包含哪些批次
                                            # 每个批次：库里行数、写出、留出、剔除、拒收（分原因）
<out>/_rejects/<批次>.jsonl                 # 没写出的行：{"id", "data_format", "task_type", "reason"}
```

- 语料类型取自库里的 `corpus_type`：book / journal / video / detection / industrial。
- `<来源>` 就是库里的 `batch_tag`，只有几个不好认的改成了可读的名字，见 `export_train_data.py` 里的 `SOURCE_NAMES`：
  - `video_v1` → `video_bilibili`（B站视频）
  - `video_abnormal_behavior_v1` → `video_park_abnormal`（园区异常行为）
  - `aircraft_vqa_v1` → `aircraft_defect`（飞机缺陷问答）
  行内 id 仍用库里的批次名。
- **qa 和 vqa 按这一行带不带媒体来分，不看 `task_kind_map`**，这样有两个好处：
  - 每个文件的字段完全一致。字段不一致时，datasets 会报 CastError。
  - vqa 里不会出现缺图的行。
- 同一批次、同一格式里如果既有图又有视频，才会出现 `.image` / `.video` 后缀。目前的数据里没有这种情况。ZeRO-3 下，一个 micro-batch 里图和视频混在一起会死锁，所以这两类必须分开。
- 每行第一个字段是 `"id": "<batch>:<库 id>"`，和评估集的 id 写法一致。所以 pt 合成一个文件后照样能追溯来源，泄漏检查也直接按 id 查。
- 行级规则沿用 9-30 那次清洗（`clean_invalid_rows.py`）：
  - 轮次必须一问一答交替，且没有空轮次；
  - 有媒体却一个占位符都没有的，在第一轮用户话前补上 `<image>` / `<video>`；
  - 占位符数和媒体数对不上的，拒收。
- `aircraft_book_v1` 里问题为空、正文在 `answer` 里的行（领域知识语料）按 pt 导出。

## 本地全量副本与同步规范

全量导出放在 `/mnt/si003010kcx0/mmdata/data_process/field_full/`：

- 以后要训练数据，**直接从这里抽**，不用再查库；
- 数据有变动时，**库一版、本地一版，同一轮完成**：
  1. 先入库或登记；
  2. 再用 `export_train_data.py --only <批次>` 加 `splice_export.py` 更新本地；
  3. 跑 `verify_export.py`、`leak_check.py` 检查；
  4. 最后跑 `summarize_export.py` 重写那里的 README。

## 一次完整流程

```bash
H=/mnt/si003010kcx0/mmdata/mmcode/eval_hk/tools/holdout
PY=/mnt/si003010kcx0/mmdata/mm_env/mysql/bin/python      # 带 pymysql
OUT=/mnt/si003010kcx0/mmdata/data_process/field_full.new      # 整体重建：导到新目录，校验通过后替换 field_full
# 1. 新评估集先登记（期刊示例），并补齐跨池副本
$PY $H/build_journal_eval.py --batch journal_gi82_v1 --set journal_gi82_eval_v1 --out <eval>.jsonl
$PY $H/register_eval_sets.py --books journal_gi82_eval_v1 journal_gi82_v1 <eval>.jsonl.issues.txt   # 确认后加 --apply
$PY $H/find_copies.py > copies.tsv && $PY $H/register_eval_sets.py --copies copies.tsv             # 确认后加 --apply
# 2. 导出（先估时间）
$PY $H/export_train_data.py --out $OUT --bench
$PY $H/export_train_data.py --out $OUT --workers 8
# 3. 检查，两项都通过才能训练
$PY $H/verify_export.py $OUT
/opt/conda/bin/python $H/leak_check.py --train $OUT --out $OUT/_logs/leak_check
$PY $H/summarize_export.py $OUT --leak $OUT/_logs/leak_check/summary.json
```

新建评估集时，在下一次导出**之前**登记。新的图片评估集还要加进 `find_copies.py` 和 `leak_check.py` 的评估集列表。
