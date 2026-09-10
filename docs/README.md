# 文档索引

根目录只留 `README.md`（工具总览与常用命令），其余按「还在用 / 设计参考 / 只作存档」分三层。

## 还在用的使用说明 —— `guides/`

| 文件 | 讲什么 |
|---|---|
| [目标检测评估_跑通流程.md](guides/目标检测评估_跑通流程.md) | 目标检测通路：一条命令跑完，跑挂了怎么缩小范围，六个坑 |
| [开放问答评估_使用说明.md](guides/开放问答评估_使用说明.md) | 装备描述问答（vqa）通路：转换 → 推理 → 裁判打分 → 报表 |
| [书籍评估_分类打分_使用说明.md](guides/书籍评估_分类打分_使用说明.md) | 书籍通路：评估集 jsonl 直读，按七大类分别出分 |
| [DPO_使用说明.md](guides/DPO_使用说明.md) | `build-dpo`：从 Alpaca/ShareGPT 直接构建 DPO 训练集 |
| [DPO_分片直接合并_使用说明.md](guides/DPO_分片直接合并_使用说明.md) | `merge_dpo_shards.py`：从已完成的推理分片直接合并出 DPO JSONL（故障恢复用） |

目标检测评估见 [../docs/目标检测评估.md](目标检测评估.md)。

## 设计文档 —— `design/`

| 文件 | 讲什么 |
|---|---|
| [独立评估工具_开发文档.md](design/独立评估工具_开发文档.md) | 这个工具为什么独立于 VLMEvalKit 的 `run.py`，模块划分与判分口径 |

## 存档 —— `history/`

已经落地的需求文档、设计文档和一次性交接说明。**只作查阅，不再维护**；当前行为以代码和
`README.md` 为准，两者不一致时以代码为准。

- `统一入口_需求文档.md` —— 统一 `convert/infer/eval/all` 入口的需求（已实现）
- `二值对照组与效应量检验_交接note.md` —— rubric v3b/v4b 与 `--dz-test` 的一次性交接说明（已应用）
- `superpowers/plans/`、`superpowers/specs/` —— 历次功能的计划与设计文档，按日期命名

## 离线跑通整条链路（不需要涉密数据）

```bash
# 数据构建端仓库里：拉公开的 VisDrone（548 图 / 38759 框）
python scripts/get_visdrone.py --out ./data/visdrone

# 本仓库：起假 VLM 顶替构建期要调的那个服务
python scripts/fake_vlm_server.py 18899 &

# 构建端 config/local.yaml 把 vlm.api_url 指过去，然后 python scripts/build.py
# 拿产出的 test.jsonl 跑 python -m eval_tool eval --config det.json
```

`scripts/fake_vlm_server.py` 产出的描述是**套模板的**，可以用来验证链路通不通、报表出不出，
**不能**用它跑出来的分数说明任何模型质量问题。

## 不在这里的东西

- 模型评估的**需求文档**（v2，目标检测）在数据构建端仓库：
  `liugenzzz/target_detection_vl_dataset` 的 `qwen3vl_sft_builder/docs/模型评估_需求文档.md`。
  本仓库的 `docs/目标检测评估.md` 是它的**实现说明**，两者分工不同。
- 训练期评估（LLaMA-Factory 集成、`vendor/VLMEvalKit`）在另一条分支上开发中，与本目录无关。
