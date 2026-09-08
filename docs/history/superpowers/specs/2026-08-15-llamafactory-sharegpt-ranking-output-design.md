# LLaMAFactory ShareGPT Ranking 输出修复设计

## 目标

项目生成的 DPO/ORPO 训练 JSONL 必须能被 LLaMAFactory 在
`formatting: sharegpt`、`ranking: true` 下识别为成对偏好样本。

## 根因

项目已经把 Alpaca 和 ShareGPT 输入统一规范化为 `conversations` 输出，
但最终序列化时仍把 `chosen` 和 `rejected` 写成字符串。LLaMAFactory 的
ShareGPT converter 只有在两者都是消息对象时才进入 pairwise 分支；字符串
会落入普通对话分支，导致单轮 `_prompt` 为空、多轮 `_prompt` 为结构体，并且
`rejected` 不参与偏好训练。

## 输出契约

每行保持且只包含四个字段：

```json
{
  "conversations": [{"from": "human", "value": "<image>\n问题"}],
  "chosen": {"from": "gpt", "value": "标准答案"},
  "rejected": {"from": "gpt", "value": "模型答案"},
  "images": ["原始图片路径"]
}
```

- `conversations` 和 `images` 的内容及顺序不变。
- `chosen/rejected` 只增加 ShareGPT 助手消息包装，不改答案文本。
- 不增加配置开关，避免未来批次再次误生成不可训练的字符串格式。
- 不删除合法的单轮、多轮、纯文本或多图片样本。

## 修改范围

- 主流水线和分片恢复脚本统一生成上述结构。
- 产物校验器拒绝旧的字符串格式，避免错误文件被发布。
- 测试覆盖 Alpaca、ShareGPT、主流水线、断点重建和分片合并。
- 使用说明同步 LLaMAFactory 的 `dataset_info.json` 配置示例。
