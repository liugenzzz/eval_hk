# 园区异常事件评估草案

对外链路名固定为 `park`。现有 `park_video_v1` 历史结果保留为通用视频问答；其行为类别是字幕关键词弱标签，不可作为异常识别真值。新异常事件是 `park` 的独立评估维度，不把旧分数改称异常识别分。本工具仅接受人工确认的异常片段，不读取现有 180 道开放题作为异常样本。

每行一段 JSONL，字段如下（路径和内容仅为格式示例）：

```json
{"taxonomy_version":"park_abnormal_v1","clip_id":"nwpu-test-001-10-15","video":"/data/nwpu/test/001.mp4","duration_s":30,"event_type":"running","interval":{"start_s":10,"end_s":15},"description":"A person runs across the campus walkway.","reviews":[{"reviewer":"annotator_a","event_type":"running","interval":{"start_s":10,"end_s":15},"description":"A person runs on the path."},{"reviewer":"annotator_b","event_type":"running","interval":{"start_s":10.2,"end_s":15.1},"description":"Someone runs on a walkway."}],"review_agreement":{"event_type":true,"interval":true,"description":true},"verification":"verified","adjudicator":"review_lead"}
```

类别闭集在 `eval_tool/park_abnormal.py` 的 `TYPE_SYNONYMS` 中，首版为奔跑、抛物、打架、摔倒、逆行、徘徊、攀爬、闯入八类。增加类别或同义词须更新 `taxonomy_version` 并复核数据。NWPU 的逐帧 0/1 标签和 Avenue 掩码只能定位候选片段，不能推导事件类别。两人独立提交类别、起止时间、描述；类型一致须与最终标签相同，时间一致允许 0.5 秒边界误差，描述一致由复核人明确判断而不按字符串相等判定。任一项有分歧，须填写独立复核人的 `adjudication_reason`，再设 `verification=verified`。需先人工核对视频授权、来源和真实事件边界。

NWPU 测试集可先生成只读复核清单：

```bash
python tools/nwpu_review_queue.py --groundtruth /path/to/NWPU_Campus_gt.npz \
  --videos /path/to/NWPUCampusDataset/videos/Test --out nwpu_review_queue.csv
```

2026-10-10 用服务器现有发布包验证：242 段测试视频与掩码帧数一一匹配，118 段掩码全为正常，另 124 段有 137 个连续异常时段。生成清单见本地 `_omni_ingest/eval_project_validation/nwpu_review_queue_20261010.csv`；其中 `event_type`、`description` 均为空，不能直接投入评分，也未生成片段剪辑。

校验：`python -m tools.park_abnormal_eval annotations.jsonl`。预测文件同样是 JSONL，包含 `clip_id`、`answer`，可带 `event_type` 和 `interval`。计分：`python -m tools.park_abnormal_eval annotations.jsonl --predictions predictions.jsonl --require-time`。该命令逐行输出自由回答的类型命中和 tIoU；可选结构化 `event_type` 只单列为 `structured_type_hit`，不能替自由回答取得类型分。描述分留空，由接入 `video_ref_v1` 裁判的调用方通过 `description_judge` 传入，三个指标独立报告。不要求报时间的题目省略 `--require-time`，缺失时间不计 tIoU。

这只是评估基础设施，尚无已复核的新异常评估集或实测分数。正式建集前须登记评估留出并通知数据端；登记会写库，需先取得用户同意。当前没有运行模型，也没有改动训练数据。
