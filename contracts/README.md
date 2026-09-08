# 边训边评 · 接口契约

训练侧（LLaMA-Factory）与评估侧（编排器）之间的接口。

> **训练侧同学从这里开始 → [`训练侧对接说明.md`](训练侧对接说明.md)**
> 那一份是自包含的：要做什么、字段逐行注释、可直接抄的 Callback 代码、
> 硬约束、自测方法、常见错误对照表、FAQ。读完那一份就够了，不必看本仓库其他文档。

写完信号文件后先自测，不用来回确认：

```bash
python3 contracts/validate_signal.py <signal_dir>/<run_id>/step-600.json
python3 contracts/validate_signal.py --dir <signal_dir>/<run_id>
```

纯标准库，不需要装任何依赖。

## 本目录文件

| 文件 | 给谁看 | 说明 |
|------|--------|------|
| `训练侧对接说明.md` | **训练侧** | 对接主文档，自包含 |
| `checkpoint_saved.example.json` | 训练侧 | 入参示例：每存一个 checkpoint 发一次 |
| `run_finished.example.json` | 训练侧 | 入参示例：训练结束发一次 |
| `validate_signal.py` | 训练侧 | 信号校验脚本，写完自己验 |
| `eval_result.example.json` | 评估侧 | **出参**示例，训练侧不用管，放这里是让双方都能看到最终数据长什么样 |

字段含义与全部边界约束另见 `../边训边评_设计文档.md` 第二、三、十一节。

## 信号目录：固定的，不跟着 output 走

```
信号目录（固定，双方约定死）
  /mnt/si003010kcx0/mmdata/mmcode/trainspace/eval_signals/

checkpoint 路径（每次训练不同，放在 JSON 内容里传）
  信号里的 checkpoint_path 字段
```

评估侧的常驻进程需要一个固定的地方去扫。信号目录如果跟着每次训练的 `output` 变，
编排器就不知道该看哪里。动态的部分全部走 JSON 字段。

## 写入约定

```
<signal_dir>/<run_id>/step-<step>.json        # 先写
<signal_dir>/<run_id>/step-<step>.json.done   # 后写，空文件
<signal_dir>/<run_id>/finished.json           # 训练结束
<signal_dir>/<run_id>/finished.json.done
```

顺序：**checkpoint 目录完全写完 → 原子写 .json → fsync → 写 .done**。

评估侧只扫 `*.done`，看到才读同名 `.json`。不要依赖目录 mtime、文件大小或
`checkpoint-*` 目录是否出现来判断保存完成——LLaMA-Factory 保存是多文件写入，
读到写了一半的目录会加载出错误权重且不报错。

## 字段规则

- **示例里出现的字段全部必填**，没有可选项。缺一个直接判 `failed`。
- 多余字段会被忽略，不报错。
- `schema_version` 不匹配直接拒收。
- `train_meta.mix_strategy` 必须是 `"concat"`；用 interleave 的话条数口径不成立，
  需要先改协议（详见设计文档 A-4）。
- `train_meta.datasets[].samples` 是**实际参训条数**：设了 `max_samples: N` 就填
  `min(N, 文件行数)`，没设就填文件行数。

## run_id

```
<模型简称>_<数据版本>_<YYYYMMDD>_<HHMM>
例：qwen3vl8b_zbv3_20260908_1430
正则：^[a-z0-9][a-z0-9_]{2,63}$
```

一次训练内固定不变；断点续训沿用同一个；不同训练不能重复。

## 训练侧硬约束

`save_total_limit` 不设或设得足够大。评估存在排队和节流，checkpoint 可能在保存后
一段时间才被评估；如果期间被轮转删除，这个点就永久缺失。存储充足，不要开轮转。
