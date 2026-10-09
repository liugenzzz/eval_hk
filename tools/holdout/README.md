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
| `export_train_data.py` | 导出训练文件，按语料类型、书籍领域 / 工业子领域、pt·qa·vqa 和任务类型分目录；同一目录同一任务合并不同来源。跳过 `is_holdout=1` 的行，以及 `--exclude-media` 里列出的媒体（兜底用）；`--bench` 先估时间。只有加 `--dataset-info` 才生成 LlamaFactory 注册文件 | 只读 |
| `splice_export.py` | 数据或剔除规则变了之后，只重导受影响的批次，再拼回原导出；所有任务文件按 id 前缀替换受影响批次，保留其他来源 | 只读 |
| `summarize_export.py` | 生成导出目录的 `README.md`：说明是全量数据、使用和更新规范、统计、对账、泄漏检查、更新记录 | 只读 |
| `eval_copies.py` | 从 md5 缓存里列出和评估图内容相同的训练图（早期做法，已被 `find_copies.py` 取代） | 只读 |
| `verify_export.py` | 校验导出结果：格式、占位符、行数对账、媒体抽查、`dataset_info` 一致性 | 只读 |
| `leak_check.py` | 递归查找全部 qa/vqa 目录，拿其中的文件和全部评估集逐行比对：答案原文、媒体路径和 md5、库 id。**每次导出后、训练前都要跑一遍**。评估集用统一名显示（aircraft_defect / video_park_abnormal / detection / industrial / book / journal / journal_gi82），括号里是库里登记名 | 只读 |
| `fielddb.py` / `mdl.py` | 数据库连接配置（从 `$FIELD_DB_CONF` 读，默认 `mm_env/field_tools/db.json`，不进仓库）/ 在共享表上安全执行 DDL | — |

## 导出产物结构

```text
<out>/book/<七大类>/{pt,qa,vqa}/<任务类型>.jsonl
<out>/book/<七大类>/pt/books_combined.jsonl   # CPT 按七大类分，不再细分，沿用原名
<out>/book/其他/<304|航空书籍|军事武器|GI82书籍|未分类>/{pt,qa,vqa}/<任务类型>.jsonl
<out>/journal/{pt,qa,vqa}/<任务类型>.jsonl    # journal_v1 + journal_gi82_v1，共 11 种
<out>/video/vqa/<任务类型>.jsonl            # B站 3 种，园区合成 3 种
<out>/detection/vqa/<场景类别>.jsonl         # 通用目标 / GQA场景 / 航拍 / 遥感
<out>/industrial/飞机缺陷/vqa/<任务类型>.jsonl  # 16 种
<out>/industrial/工业质检/vqa/defect_cn.jsonl   # part0 + part1 中文合并
<out>/industrial/工业质检/vqa/defect_en.jsonl   # part0 + part1 英文合并
<out>/README.md                            # summarize_export.py 生成
<out>/manifest.json                        # 文件：领域、任务、行数、字节、格式、模态、来源批次
                                          # 批次：库里行数、写出、留出、剔除、拒收（分原因）
<out>/_rejects/<批次>.jsonl                 # {"id", "data_format", "task_type", "reason"}
```

- 语料类型取自库里的 `corpus_type`：book / journal / video / detection / industrial。
- 七大类为 `domain_code` 01～07 对应的共性基础、研发设计、生产制造、试验测试、维修保障、运营管理、作战应用。没有七大类标签的书籍按其他标签归档。
- 文件名按任务类型命名；同一领域、同一任务的不同来源合并。检测例外：按四种场景类别各一个文件。
- 园区 24 个子集按题型合成 `park_caption.jsonl`、`park_multiple_choice_qa.jsonl`、`park_open_ended_qa.jsonl`；B站保留 `temporal_reasoning.jsonl`、`video_caption.jsonl`、`video_vqa.jsonl`。
- pt 每行是 `{"id", "text"}`。其他格式统一转成 sharegpt：openai 的角色转为 human / gpt；alpaca 的 instruction 和 input 合成 human，output 为 gpt；pipeline_export 用 `train_conversations`。
- pt / qa / vqa 按 `task_kind_map` 定，带图或视频的行一律进 vqa。vqa 每行带 `images` 或 `videos`，缺媒体写空列表，保持字段一致。
- 图和视频分开导出，必要时用 `.image` / `.video` 后缀，避免训练 micro-batch 混合媒体。
- 每行第一个字段是 `"id": "<batch>:<库 id>"`，合并后仍可追溯来源；manifest 的 `batches` 也保留库里的批次名。
- 来源显示名 `SOURCE_NAMES` 对照保留：`video_v1` → `video_bilibili`（B站视频）、`video_abnormal_behavior_v1` → `video_park_abnormal`（园区视频）、`aircraft_vqa_v1` → `aircraft_defect`（飞机缺陷）。v3 不再使用 `rename_sources.py` 改任务文件名。
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
# 2. 导出（先估时间；长任务放 tmux / nohup 中后台运行，日志写文件）
$PY $H/export_train_data.py --out $OUT --bench
$PY $H/export_train_data.py --out $OUT --workers 8
# 3. 检查，两项都通过才能训练
$PY $H/verify_export.py $OUT
/opt/conda/bin/python $H/leak_check.py --train $OUT --out $OUT/_logs/leak_check
$PY $H/summarize_export.py $OUT --leak $OUT/_logs/leak_check/summary.json
# 4. 验收行数、目录及泄漏结果后替换；备份目录必须尚不存在
BASE=/mnt/si003010kcx0/mmdata/data_process
test ! -e "$BASE/field_full.v2_old"
mv "$BASE/field_full" "$BASE/field_full.v2_old"
mv "$OUT" "$BASE/field_full"
# 正式目录改名后重写 README，让位置及所有命令指向正式路径
$PY $H/summarize_export.py "$BASE/field_full" --leak "$BASE/field_full/_logs/leak_check/summary.json"
# 旧版备份由用户自行删除
```

新建评估集时，在下一次导出**之前**登记。新的图片评估集还要加进 `find_copies.py` 和 `leak_check.py` 的评估集列表。
