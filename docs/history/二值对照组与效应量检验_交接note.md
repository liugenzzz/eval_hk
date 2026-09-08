# 二值对照组 + 效应量检验

## 覆盖这几个文件

```
eval_tool/judge_rubrics.py     # 加了 v3b / v4b 解析
eval_tool/aggregate.py         # 上一轮的 total_score 修复（如果还没覆盖）
compare_rubrics.py             # 实装了 --dz-test
gen_rubric_prompts.py          # 六版一起生成，preamble 逐字节相同
prompts/judge_equip_pointwise_v3b.txt
prompts/judge_equip_pointwise_v4b.txt
```

`judge.py` / `score_vqa.py` / `run_eval.py` / `report.py` **不用再动**，上一轮那版就够了。

## 第一步：先免费拿到「后置阈值」那两组，不用重跑

你 v3 / v4 已有的 detail 表里，`hit` 列本身就是 `quality_score >= 60`。所以后置二值化
的结果现在就能算出来：

```bash
python compare_rubrics.py --baseline base --dz-test --metric hit \
    v3=runs/v3/judge_detail_all.xlsx \
    v4=runs/v4/judge_detail_all.xlsx
```

和不加 `--metric hit` 的那次对比，差值就是**报表阈值化损失掉的分辨力**。

## 第二步：跑原生二值版（需要重新判分）

```json
"pointwise": "prompts/judge_equip_pointwise_v3b.txt"
```

跑完换 v4b 再跑一次。缓存不用手动删，fingerprint 会自己失配。

## 第三步：四组一起比

```bash
python compare_rubrics.py --baseline base --dz-test \
    v3=runs/v3/judge_detail_all.xlsx   \
    v4=runs/v4/judge_detail_all.xlsx   \
    v3b=runs/v3b/judge_detail_all.xlsx \
    v4b=runs/v4b/judge_detail_all.xlsx
```

`--dz-test` 会重采样共享的 index 集合，对每一组算 d_z，再看两两之差的置信区间跨不跨 0。
跨 0 = 这两组在你这批数据上分不出高下，只能靠人工对齐集决定。

## 预期

二值组的 `distinct_values` 会是 2，平局率高，d_z 大概率低于连续组 —— 这是预期结果，不是 bug。
真正值得看的是：**二值组在跟人工的一致性上会不会反超**。粗判定往往单次更可靠，
所以这个 2x2 的价值在于把「分得开」和「判得对」拆成两件事分别看。

## aspect_failures

v3b / v4b 除了 `correct` 还输出 `aspect_failures` 数组（param / fact / visual /
fabrication / relevance），落到 detail 表的 `failure_type` 列，逗号分隔。
判 false 但没给归因的行会记成 `unattributed` —— 这个比例高说明裁判没听归因指令。
