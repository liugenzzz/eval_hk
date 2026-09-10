# 独立多模型 VLM 评估打分工具

把一批模型在评估集上的**推理预测**离线打分，产出**多模型对比报告**。推理和判分分开：推理很贵、判分相对便宜，所以预测文件跑一次就固化下来，加模型、换口径、换 rubric 都只重跑判分，不重复推理。

> 文档索引见 **[docs/README.md](docs/README.md)**：使用说明在 `docs/guides/`，设计文档在 `docs/design/`，已落地的需求与交接说明存档在 `docs/history/`。

---

## 这个工具现在评三样东西

| 领域 | 评什么 | 数据集 | 打分方式 | 详细文档 |
|---|---|---|---|---|
| **装备** | 航空装备的选择题、判断题、开放问答 | `aero_mcq` / `aero_judge` / `aero_vqa` | 选项字母抽取（代码）+ 看图裁判（pointwise / pairwise） | [开放问答评估_使用说明](docs/guides/开放问答评估_使用说明.md) |
| **书籍** | 同上，换一批书籍领域的数据 | 自己命名，如 `book_mcq` / `book_vqa` | **与装备完全相同** —— 同一套 kind、同一套提示词、同一套报表维度 | 同上 |
| **目标检测** | 指代定位、多框检测、物体识别、计数、清单、拒答、区域描述 | `eval_set_v1`（构建端产出的 `test.jsonl`） | 九个纯代码打分器 + 一个裁判打分器 | **[目标检测评估](docs/目标检测评估.md)** |

三个领域跑的是**同一套代码**，区别只在配置。装备和书籍连提示词都共用，只是数据不同；目标检测多了一批新的打分器和报表维度。

---

## 骨架：数据集声明 kind，打分器按 kind 查注册表

这是三个领域共用的机制，也是加第四个领域时唯一要动的地方 —— **写配置，不改代码**。

`datasets` 的每个条目声明自己是什么答案形态（`kind`），`run_eval` 按 kind 查注册表分派打分器：

```json
"datasets": {
  "mcq":     "aero_mcq",
  "book_tf": { "name": "book_tf",     "kind": "choice",
               "params": { "choice_style": "judge" } },
  "ground":  { "name": "eval_set_v1", "kind": "grounding_single",
               "params": { "iou_gate": 0.5, "dev_threshold_pct": 5.0 } }
}
```

- 字符串是旧写法，等价于只写 `name`。`mcq` / `judge` / `vqa` 三个历史键名不写 `kind` 时分别回落到 `choice` / `choice` / `judge_text`，**旧配置照跑**。
- **其他任何键名都必须显式声明 `kind`**。猜错打分器会静默出一份错的报表，比直接报错难查得多。
- `kind` 未实现时在**起跑前**报错，不会先跑掉几小时推理或裁判调用。
- `params` 传给该打分器，口径参数写在这里，不写死在代码里。

### 已实现的 kind

| kind | engine | 用在哪 | 说明 |
|---|---|---|---|
| `choice` | code | 装备 / 书籍 | 选择题 / 判断题。`params.choice_style="judge"` 时按 A/B 二值判，否则 A/B/C/D |
| `judge_text` | judge | 装备 / 书籍 | 自由文本，裁判 pointwise 打分，并参与 base vs sft 的 pairwise |
| `grounding_single` | code | 目标检测 | 单框定位。达标率 + 定位成功率 + 四点偏差 + 有符号 bias |
| `grounding_multi` | code | 目标检测 | 多框检测。匈牙利匹配后算 P/R/F1、数量准确率与误差、框级达标率 |
| `object_ident` | code | 目标检测 | 物体识别。精确 / 上位 / 下位 / 错误四档，表外词交裁判兜底 |
| `short_answer` | code | 目标检测 | 短答案，归一化精确匹配，不中的交裁判兜底 |
| `counting` | code | 目标检测 | 计数。精确命中率 + MAE + 偏向；`counting=="zero"` 一路并进拒答表 |
| `inventory` | code | 目标检测 | 清单。类别集合 P/R/F1 与数量分开判，不合成一个分 |
| `exist_negative` | code | 目标检测 | 拒答表。拒答准确率 + yes 偏置率 |
| `describe` | judge | 目标检测 | D 组描述。代码判范围合规 / CHAIR / 空话，裁判判正确性 / 落地性 / 信息量 |
| `perturbation` | code | 目标检测 | 问法扰动。按组算三次输出之间的 IoU 一致率和四点方差 |

`engine` 区分「代码打分」和「裁判打分」：`engine="code"` 的打分器**不调任何模型**，同一份预测重跑一百遍逐位相同；`engine="judge"` 的结果带裁判噪声，只适合纵向对比。验收总分只由代码组构成。

加一种新的答案形态，写一个打分函数挂到注册表上即可，`run_eval` 不需要改：

```python
# eval_tool/scorers/my_kind.py
from . import CODE, ScoringContext, register

@register("my_kind", engine=CODE, answer_form="text")
def score_my_kind(data, ctx):
    ...
```

---

## 统一入口

复制 `pipeline.example.json` 为 `pipeline.json`（目标检测用 `det.example.json`），在一份配置里填数据、多个模型、推理和裁判参数：

```bash
python -m eval_tool convert data.json --config pipeline.json   # ShareGPT JSON -> TSV
python -m eval_tool infer   --config pipeline.json             # 推理，断点续传
python -m eval_tool eval    --config pipeline.json --rubric v4 # 判分出报表
python -m eval_tool sweep   --config pipeline.json --rubrics v1,v3,v4,v4b
python -m eval_tool all     --config pipeline.json             # 一次跑完
```

- 推理默认断点续传；只有显式 `--overwrite` 才会从头跑。推理指纹变化时会报错，不会自动触发几小时的 GPU 重跑。
- 每个 batch 写入 `work_dir/<model>/_partial/*.jsonl`；默认保留以便核查，完成后可用 `--clean-partial` 清理。
- 评估输入优先级为 `models[].scored > models[].pred > 约定推理路径`。`sweep` 为了保证 rubric 可验证，拒绝使用 `scored`。
- 多模型和多 rubric 按配置顺序串行。`sweep` 的比较表会覆盖每个 challenger，不只是第一个。

旧入口仍保留：`python -m eval_tool.run_infer --config infer_config.json`、`python -m eval_tool.run_eval --config config.json`，以及无子命令的 `python -m eval_tool --config config.json`。

---

## 装备评估

这是最早的那条通路，三个数据集：

| 数据集键 | 真值文件 | kind | 打分 |
|---|---|---|---|
| `mcq` | `aero_mcq.tsv` | `choice` | 从模型输出里抽 A/B/C/D，和金标比 |
| `judge` | `aero_judge.tsv` | `choice` | 同上，但只有 A/B 两个选项 |
| `vqa` | `aero_vqa.tsv` | `judge_text` | 看图裁判 pointwise 打分，再做 base vs sft 的 pairwise（含位置互换） |

裁判 rubric 有六个版本，外置在 `prompts/judge_equip_pointwise_v*.txt`：

```
v1  v2  v3  v3b  v4  v4b
```

`--rubric v4` 指定一版；`--rubric` 换版本会**换掉判词缓存的指纹**，自动重判而不是拿旧判词冒充新口径。`sweep` 一次跑多版并出比较表，用来看 rubric 之间一致不一致 —— 一致率低说明 rubric 有歧义，**先改提示词再看分数**，否则是在解读噪声。

`score_summary.csv` 的 `total_score` 按 `category_weights` 对六个能力维度（P1/P2/P3/R1/R2/R3）加权。样本数少于 30 的维度不进总分但单独出行 —— 一个 n=2 的类别拿满分不该把总分拉上去。

开放问答那条路的完整用法（数据转换、长度控制、成对判定）见 **[docs/guides/开放问答评估_使用说明.md](docs/guides/开放问答评估_使用说明.md)**。

---

## 书籍评估

**和装备是同一套东西，只是换一份数据。** 提示词、kind、报表维度、加权口径全部共用，没有任何书籍专属的代码。

配置写法就是给数据集起个别的名字，然后显式声明 kind：

```json
"datasets": {
  "book_mcq": { "name": "book_mcq", "kind": "choice" },
  "book_tf":  { "name": "book_tf",  "kind": "choice",
                "params": { "choice_style": "judge" } },
  "book_vqa": { "name": "book_vqa", "kind": "judge_text" }
},
"enabled_datasets": ["book_mcq", "book_tf", "book_vqa"]
```

两点注意：

- 判断题（两个选项）要写 `params.choice_style: "judge"`。历史上这件事是靠数据集**键名叫不叫 `judge`** 来判的，新键名不会自动继承那个规则。
- 装备和书籍放在同一份配置里跑也可以，报表里按 `dataset` 列天然分开。想分别加权就在 `report.dataset_weights` 里各给一个系数。

### 评估集自带领域分类时，按分类分别出分

书籍那批评估集每条自带一个领域分类（作战应用、拱形基础……七大类），要按这七类分别出分。
评估集是 jsonl 就不用转 TSV —— 真值按数据集名找，**优先 `<name>.jsonl` 再回落 `<name>.tsv`**，
`conversations`（ShareGPT）和 `messages`（OpenAI）两种形状都认。配置两处：

```json
"datasets": {
  "book_vqa": { "name": "book_vqa", "kind": "judge_text",
                "params": { "category_field": "category",
                            "image_root": "/评估机器上的/images" } }
},
"report": { "dims": [ { "key": "category", "from": "category", "min_n": 10 } ] }
```

`params.category_field` 把记录里那个字段放到 `category` 列（不写就还是 `metadata.task_type`），
`report.dims` 让 `breakdown.csv` 按它拆行。`metric=quality_score` 那几行是裁判 rubric 的
0-100 原分（「作战应用 80 分」要的就是它），`metric=hit` 是通过率 —— 通过率一样的两个分类
均分可以差十几分，两个口径别混着看。

分类多而每类样本少时记得放宽两个门槛：维度上的 `min_n`（管 `breakdown.csv` 标不标灰）和
顶层的 `min_category_n`（管 `score_summary.csv` 的 `total_score` 算不算这一类），默认都是 30。

只有 `{"text": ...}` 的语料行没有问答对，会被跳过并计数 —— 要评估这部分内容得先转成问答形式。

完整用法见 **[docs/guides/书籍评估_分类打分_使用说明.md](docs/guides/书籍评估_分类打分_使用说明.md)**，
示例配置是 `book.example.json`。

---

## 目标检测评估

新增的那条通路，被测对象是 Qwen3-VL-8B-Instruct 的 SFT checkpoint（对比 base），数据是构建端（`liugenzzz/target_detection_vl_dataset`）产出的 `test.jsonl`。

这次评估要回答三个问题：SFT 之后指代定位的能力涨了多少、涨的部分是不是靠牺牲别的能力换来的、**哪一档哪一类哪种尺寸的目标还不行**。第三点是真正的产出 —— 只出一个总分没有用。

十一个数据集（八个主线 + 三个派生），完整配置见 **`det.example.json`**：

| 数据集 | kind | 考什么 |
|---|---|---|
| `ground_box` | `grounding_single` | 7 个 `ground_*` 的轮 1 + `inventory_locate` 的轮 2 |
| `detect_box` | `grounding_multi` | `detect_class` / `detect_describe` 的轮 1 |
| `region_identify` | `object_ident` | 框 → 类别 |
| `count_class` | `counting` | 数量 |
| `inventory` | `inventory` | 全图清单 |
| `attribute_qa` | `short_answer` | 属性短答案 |
| `exist_negative` | `exist_negative` | 拒答 |
| `describe` | `describe` | 各任务的描述轮 |
| `describe_modelhist` | `describe` | 历史轮换成模型自己的输出 → 链路衰减率 |
| `reverse_consistency` | `object_ident` | 正向的框反问「这框里是什么」 |
| `question_perturbation` | `perturbation` | 同一目标换 3 种问法 |

画框的验收指标：

```
达标(单样本) = 解析出框 ∧ IoU ≥ iou_gate ∧ 四点平均偏差 / scale ≤ dev_threshold_pct
达标率 = 达标样本数 / 全部样本数            默认 iou_gate=0.5、dev_threshold_pct=5.0
```

不达标的拆三个桶，加起来 = 1 − 达标率：`malformed`（格式不合规，训练配置问题）/ `localize_fail`（框到别的目标，补指代消歧和密集场景）/ `deviation`（框对了不够准，看 `bias_*` 定方向）。

**跑之前必配三个路径**。十一个数据集用的是同一份图和同一张类别表，所以写在顶层 `dataset_defaults` 一处，自动铺给每个数据集（数据集自己写的同名键仍然优先）：

```json
"dataset_defaults": {
  "image_root":   "/评估机器上的/images",         ← 推理端一律要图
  "classes_yaml": "/评估机器上的/classes.yaml",   ← 不配 E 组主指标会偏高
  "labels_dir":   "/评估机器上的/labels"          ← 配了 CHAIR 才有数
}
```

`image_root` 不配的话推理时模型看不见图，框出来的全是废的。评估端会自动跳过图片编码 —— 代码打分器不看图，只有裁判组才读。推理提示词同理：这十一个数据集都是原样透传问句，写一处 `infer.prompt_file`（单数）铺给全部，不用把同一行抄十一遍。

**有第二个裁判就配上交叉验证**。需求文档自己写了一条做不到的局限：裁判和被测同家族，没法做自偏检测 —— 分高到底是模型强还是裁判认亲，一个裁判分不出来。配一路异家族裁判就补上了：

```json
"judge": {
  "model": "qwen3.6-27b",
  "cross_check": [{"name": "internvl", "api_base": "http://127.0.0.1:18181/v1/chat/completions", "model": "InternVL2-26B"}]
}
```

每路只写和主裁判不同的字段，其余（尤其是提示词）全部继承。主裁判出的 `hit` 一个数都不变，明细表每行多一列 `hit__internvl`，报表多出 `judge_agreement.csv`（逐行一致性）和 `judge_conclusion.csv`（**换裁判之后结论会不会翻**）。后者是非看不可的那张：`verdict=flipped` 说明这份增益多半是裁判的家族偏好，不能报。

**跑法就一条命令**：`python -m eval_tool all --config det.json`（推理 → 造派生集 → 再推理 → 评估 → 报表）。要改的四处配置、跑完看哪几张表、以及跑挂了怎么缩小范围，见 **[docs/guides/目标检测评估_跑通流程.md](docs/guides/目标检测评估_跑通流程.md)**。

其余全部细节 —— 打分器口径、评估集 `test.jsonl` 直读、D 组代码判与裁判判的分工、派生评估集、四指纹冻结、报表拆分与标灰规则 —— 见 **[docs/目标检测评估.md](docs/目标检测评估.md)**。

---

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
  "image_min_pixels": 65536,
  "image_max_pixels": 589824,
  "device_map": "auto",
  "gpu_ids": [],
  "workers_per_gpu": 1
}
```

- `batch_size`：单个模型实例一次处理多少条。显存够可以调大，比如 2、4、8。
- `image_min_pixels` / `image_max_pixels`：交给 checkpoint processor 前的总像素面积下界/上界，规则见上文“图像像素面积与训练配置对齐”。
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

- `enabled_datasets`：只写你要跑的数据集 key。不在列表里的数据集完全不加载真值，也不要求模型配置对应的预测路径。默认是 `datasets` 里声明的全部。
- `category_weights`：装备/书籍六个能力维度（P1/P2/P3/R1/R2/R3）的加权系数，用于 `score_summary.csv` 的 `total_score`。样本先按 `category` 汇总成每个维度的原始分，再加权平均。缺省全部 1.0；没有对应训练任务的维度可以调低，之后再调回去。只影响 `score_summary.csv`。

目标检测用的是另一套加权：`report.dataset_weights` 按**数据集**加权，产出 `acceptance_score.csv`，且只统计 `engine="code"` 的数据集 —— 裁判打的分会抖，做验收不合适。两套互不影响。

---

## Prompt 文件

推理 prompt 和裁判 prompt 全部外置在 `prompts/`，改提示词不用动代码：

```text
prompts/
  infer_mcq.txt  infer_judge.txt  infer_vqa.txt        推理（装备 / 书籍共用）
  infer_vqa_equip.txt  infer_vqa_raw.txt

  judge_equip_pointwise_v1..v4b.txt                    装备 / 书籍的裁判 rubric，六版
  judge_equip_pairwise.txt                             成对判定
  judge_vqa_pointwise.txt  judge_vqa_pairwise.txt      早期通路

  judge_describe_v1.txt                                目标检测 D 组描述，三维度
  judge_synonym_v1.txt                                 C / E 组同义兜底

  judge_dpo_*.txt  judge_category_classify.txt         DPO 构建器用
```

推理 prompt 支持用 `{question}`、`{A}`、`{B}`、`{C}`、`{D}`、`{answer}`、`{category}`、`{l2-category}`、`{source_id}` 等 TSV 列名占位。

**改推理 prompt 要重跑 `run_infer`；改裁判 prompt 只需重跑 `eval`** —— 判词缓存的键里含裁判指纹（模型 + 温度 + 两份提示词的哈希），换了提示词自动重判，不会拿旧判词冒充新口径。

目标检测还会读**数据构建端**的两类提示词文件（只读不改）：`prompts/describe/*.txt` 的 `#! must-not:` 词表用于范围合规检查，`prompts/<task>/*.txt` 的问法池用于问法扰动。路径在配置里给。

---

## 输入

真值文件放在 `tsv_dir` 下，按数据集名找，**优先 `.jsonl` 再回落 `.tsv`**：

| 领域 | 文件 | 格式 |
|---|---|---|
| 装备 | `aero_mcq.tsv` / `aero_judge.tsv` / `aero_vqa.tsv` | TSV，列见下 |
| 书籍 | 自己命名，如 `book_mcq.tsv` | 同上 |
| 目标检测 | `eval_set_v1.jsonl` | 构建端产出的 `test.jsonl`，直接读，按轮次拆行 |

TSV 需要的列：`index`、`question`、`answer`，选择题另需 `A`/`B`/`C`/`D`，可选 `image`（base64）、`history`、`category`、`l2-category`、`source_id`。

jsonl 走评估集通路：metadata 扁平化成 `meta.*` 列，多轮记录按轮次拆成多行，用 `params.select` 声明要哪些任务的哪一轮。详见 [docs/目标检测评估.md](docs/目标检测评估.md)。

预测文件每个模型每个数据集一份，至少包含：

- `index`
- `prediction`，也兼容 `pred`、`response`、`model_answer`、`answer_pred`、`模型回答`、`预测`

---

## 报表文件

新旧两套并存，旧的那套装备评估一直在用，新的那套是目标检测加的（对装备/书籍同样有效）。

| 文件 | 内容 |
|---|---|
| `score_summary.csv` | 每个模型一行，`total_score`（按 `category_weights` 加权）+ 各维度原始分与样本数。跑完直接打印在终端 |
| `report_summary.csv` / `.json` | 模型横向总表，含 CI 和长度控制列 |
| `report_summary_long.csv` | 长格式指标表 |
| `cross_{model}.csv` | 能力 × 内容类型交叉表 |
| `detail_{model}_{dataset}.xlsx` | 逐条明细，不写入 base64 图片 |
| `pairwise_vs_baseline.csv` | 各模型 vs 基准的胜/平/负率 |
| **`breakdown.csv`** | 模型 × 数据集 × 维度 × 取值 × 指标，每格带 `n` / CI / `status`。指标含裁判的 `quality_score`（0-100 原分）和 `hit`（通过率） |
| **`failure_buckets.csv`** | 达标率 + 三个失败桶 |
| **`acceptance_score.csv`** | 验收总分，只由 `engine="code"` 的数据集加权构成 |
| **`paired_diff_vs_baseline.csv`** | 相对 base 的**配对** bootstrap 差值与区间 |
| **`chain_decay.csv`** | 链路衰减率（gold 历史 vs 模型历史） |
| **`judge_agreement.csv`** | 主裁判与异家族裁判逐行一致性（`delta` / `mad` / `agree_rate` / `spearman`）。配了 `judge.cross_check` 才有 |
| **`judge_conclusion.csv`** | 换裁判之后相对 base 的增益会不会变号（`agree` / `flipped`）。同上 |
| **`run_fingerprint.json`** | 评估集哈希 / 打分口径版本 / rubric 版本 / 裁判模型 |
| `warnings.log` | 缺预测、额外 index、跳过项等警告 |

`breakdown.csv` 的 `status` 分五种，后三种在报表上都是空白 —— **实现的人看到空格会当 bug 修，所以必须分开标注**：

| status | 含义 |
|---|---|
| `ok` | n 够，可以下结论 |
| `trend_only` | 维度声明了 `level: "group"`（如难度档在 task 级），只看趋势 |
| `insufficient` | n < `min_n`（默认 30），显示 n 但不显示百分比 |
| `by_design` | 该组合**本就不产样本**（`ground_part` 没有 hard 档） |
| `not_in_data` | 该任务**产出为 0**（`ground_unique` / `spatial_relation`） |

缓存默认写到 `cache_dir`（`judge_cache_pointwise.jsonl` / `judge_cache_pairwise.jsonl`）。**缓存不要删**，重跑和加模型会复用它。

---

## 图像像素面积与训练配置对齐

> 想让「推理像素面积必须等于训练配置」在**配置加载时硬校验**（不一致直接报错，而不是只写在文档里），见 [docs/目标检测评估.md 的「推理像素面积与训练配置的硬校验」](docs/目标检测评估.md#推理像素面积与训练配置的硬校验)。

`infer_config.example.json`、`pipeline.example.json` 和 `dpo.example.json` 的 `infer` 块都显式配置：

```json
{
  "image_min_pixels": 65536,
  "image_max_pixels": 589824
}
```

这两个值限制的是 `宽 × 高` 的总像素面积，不是单独的宽或高。两项必须同时省略、同时为 `null`，或同时设为正整数，并满足 `image_min_pixels <= image_max_pixels`。省略或同时为 `null` 时维持旧推理行为。示例值与当前 LLaMAFactory `0.9.5.dev0` 训练配置一致；启用后，本项目在 checkpoint 自带的 Hugging Face processor 之前按固定的 `llamafactory-0.9.5-qwen-static-v1` 顺序预缩放，checkpoint processor 随后仍会继续执行自身处理。

该设置覆盖旧版 TSV/base64 普通推理、统一 pipeline 推理，以及 DPO 构建器的本地生成模型路径；它不改变 DPO Judge 收到的原始图片。降低 `image_max_pixels` 通常可减少视觉 token 和显存占用，但可能丢失细节；提高 `image_min_pixels` 会放大小图。

更改任一值都会改变 inference fingerprint/断点身份。统一入口需要显式运行 `python -m eval_tool infer --config pipeline.json --overwrite`，或改用新的 `work_dir`；旧 `run_infer` 入口没有对应 CLI 开关，应在 `infer_config.json` 中设置 `"overwrite": true`，或改用新的 `out_dir`；DPO 使用 `python -m eval_tool build-dpo --config dpo.json --overwrite`，或改用新的 `work_dir`。

源图生成和推理期像素规整是两层独立变换。代码审计确认，仓库中的 JPEG quality 90、最长边 1280 逻辑实际位于 `select_p123_data.py::encode_image`；`aero_vqa_dataset .py` 本身不执行该压缩。离线步骤如果已经改变源图字节，`image_min_pixels` / `image_max_pixels` 不会撤销或替代它。要达到逐像素一致，还必须使用相同源图字节、Pillow/Transformers 版本，以及相同模型 checkpoint processor 配置。

---

## 独立 JSON/JSONL → DPO 数据构建器

`build-dpo` 是一条独立通路：直接读取 Alpaca/ShareGPT 的 JSON 或 JSONL，以原始标准答案为 `chosen`，让本地 Qwen/VLM 生成 `rejected`，最终发布严格 ShareGPT DPO JSONL。它不先转成 TSV/XLSX，也不改变上面的旧评估入口。

复制并修改示例配置（示例密钥只是占位符，真实密钥不要提交到仓库）：

```powershell
Copy-Item dpo.example.json dpo.json
python -m eval_tool build-dpo --config dpo.json
```

可混合传入多个文件；解析器按文件内容识别 JSON 数组、单个 JSON 对象或 JSONL，而不依赖扩展名。命令行重复的 `--input` 会整体替换配置里的 `inputs`，不会追加：

```powershell
python -m eval_tool build-dpo `
  --config dpo.json `
  --input data/alpaca_train.json `
  --input data/sharegpt_train.jsonl
```

支持的全部命令行开关：

- `--config PATH`：必填的严格 JSON 配置。
- `--input PATH`：可重复；只要出现，就以这些输入替换配置中的 `inputs`。
- `--dry-run`：完成加载、归一化、去重和图片读取/MIME/哈希预检，不加载生成模型，也不调用 Judge。
- `--overwrite`：当输入、图片、模型、checkpoint 或生成身份变化时，创建并原子切换到新 attempt；旧 attempt 不会先被原地删除。
- `--clean-partial`：仅在最终产物成功发布后清理可再生的中间缓存。

Alpaca 多轮、多图示例；每个历史问答和当前问答都会各自展开为一条候选，当前轮只能看到此前标准历史：

```json
{
  "history": [["<image>\n描述第一张图。", "第一张图的标准答案。"]],
  "instruction": "<image>\n比较第二张图与第一张图。",
  "input": "请只回答关键差异。",
  "output": "当前轮标准答案。",
  "images": ["images/first.png", "images/second.png"]
}
```

ShareGPT 多轮、多图示例：

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\n第一轮问题"},
    {"from": "gpt", "value": "第一轮标准答案"},
    {"from": "human", "value": "<image>\n第二轮问题"},
    {"from": "gpt", "value": "第二轮标准答案"}
  ],
  "images": ["images/first.png", "images/second.png"]
}
```

配置中的关键模式：

- `wrong_only: false`：完全跳过 Judge，所有通过输入/图片校验且生成出非空、非相同答案的候选都可进入 DPO；此模式下 `rubric` 和 `judge` 即使残留也会被忽略。
- `wrong_only: true`：必须设置 `rubric: "binary"` 或 `rubric: "v4"`，并提供完整 `judge`。只有 Judge 明确认定模型答错的候选会被选择；请求或解析失败不会被当作答错。
- `infer.enable_thinking` 默认是 `false`。`model_path` 指向本地 HuggingFace 模型/checkpoint；`batch_size`、`gpu_ids`、dtype、device map 等影响生成身份，`workers_per_gpu` 仅影响调度和断点重分片。
- `judge.api_key` 必须在本地配置中替换占位值，但不要把真实凭据提交到版本库。

成功后，`output_dir` 中包含配置的训练文件名，以及 `audit_records.jsonl`、`rejected_records.jsonl`、`summary.json`、`warnings.log` 和最后发布的 `manifest.json`。训练行严格只有 `conversations`、`chosen`、`rejected`、`images`。`images` 保留输入中的原始路径字符串；解析后的绝对路径只用于内部读取和校验。

断点状态位于 `work_dir`，推理、Judge 原始响应和 Judge 解析结果分层缓存。失败且最终选择数为零时，不覆盖已有成功产物，诊断写到 `work_dir/failed_runs/<run_id>/`。当前自动化验收使用离线 fake 生成器/Judge，未执行真实 GPU 模型与真实 Judge 服务冒烟；部署环境可在配置好私有服务后另行验证。

> **评估装备描述开放问答(ShareGPT JSON 数据、只跑 vqa)请看《[docs/guides/开放问答评估_使用说明.md](docs/guides/开放问答评估_使用说明.md)》**,那是当前主用通路。本 README 描述的是原有 mcq/judge/vqa 三数据集流程,部分提示词/配置示例(P1-R3 能力分类等)不适用于开放问答通路。

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

---

## 开发

```bash
pip install -r requirements.txt
python -m pytest tests -q          # 1005 个测试
```

不需要涉密数据就能把整条链路跑一遍（拉公开的 VisDrone + 假 VLM 顶替构建期要调的服务），步骤见 [docs/README.md](docs/README.md) 的「离线跑通整条链路」。

`tests/test_legacy_regression_lock.py` 是装备 / 书籍那条老通路的**行为锁**：端到端跑完 mcq + judge + vqa，钉死选项集区分、detail 列、`score_summary` 的类别门、裁判判词缓存键、pairwise 输出列。改共用代码碰坏了它会立刻红。

---
