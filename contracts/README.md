# 边训边评 · 接口契约

本目录是训练侧（LLaMA-Factory）与评估侧（编排器）之间的接口示例。
字段含义、必填性和写入顺序见 `../边训边评_设计文档.md` 第二、三节。

| 文件 | 方向 | 说明 |
|------|------|------|
| `checkpoint_saved.example.json` | 训练 → 评估 | 每保存一个 checkpoint 发一次 |
| `run_finished.example.json` | 训练 → 评估 | 训练结束发一次 |
| `eval_result.example.json` | 评估 → 报告/看板 | 每次评估产出一份 |

## 训练侧写入约定（必须遵守）

```
<signal_dir>/<run_id>/step-<step>.json        # 先写
<signal_dir>/<run_id>/step-<step>.json.done   # 后写，空文件
```

顺序：**checkpoint 目录完全写完 → 写 .json → fsync → 写 .done**。

评估侧只扫 `*.done`，看到才读同名 `.json`。不要依赖目录 mtime、文件大小或
`checkpoint-*` 目录是否出现来判断保存完成——LLaMA-Factory 保存是多文件写入，
读到写了一半的目录会加载出错误权重且不报错。

结束信号写到 `<signal_dir>/<run_id>/finished.json` + `finished.json.done`。
