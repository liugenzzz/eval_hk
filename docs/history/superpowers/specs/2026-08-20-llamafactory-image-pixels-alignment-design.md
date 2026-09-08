# LLaMAFactory 图像像素对齐设计

## 背景与目标

训练使用 LLaMAFactory `0.9.5.dev0`、Qwen3-VL，并显式配置：

```yaml
image_min_pixels: 65536
image_max_pixels: 589824
```

评估项目目前把图片解码并转成 RGB 后直接交给 Hugging Face `AutoProcessor`，没有执行 LLaMAFactory 的应用层图像规整。目标是在所有本地 Qwen 图像推理入口中提供同名配置，并复刻 LLaMAFactory 0.9.5 的静态图像处理顺序，使训练和评估进入模型 processor 前的图像一致。

## 已确认的上游语义

LLaMAFactory 0.9.5 不会把这两个字段作为 Hugging Face `min_pixels` / `max_pixels` 传给 `AutoProcessor`。它把字段挂在顶层 processor 上，由 `Qwen3VLPlugin` 在调用模型 image processor 前执行 PIL 预处理。

Qwen3-VL 静态图像路径的顺序为：

1. 若 `width * height > image_max_pixels`，按 `sqrt(max / area)` 等比缩小，宽高使用 `int()` 截断，并调用 Pillow 默认 `resize`。
2. 若缩放后的面积小于 `image_min_pixels`，按 `sqrt(min / area)` 等比放大，同样使用 `int()` 和 Pillow 默认 `resize`。
3. 非 RGB 图像转换为 RGB。
4. Qwen3-VL 继承 Qwen2-VL 的约束：两边至少 28 像素；宽高比超过 200 时把长边限制为短边的 180 倍。
5. 将规整后的图像交给 checkpoint 自带的 Hugging Face image processor，调用时不传额外像素边界。

`0.9.5.dev0` 覆盖多个开发提交，但首个该版本提交、v0.9.5 标签和当前主线的静态图像算法一致。最终 Hugging Face patch 对齐仍由评估所加载 checkpoint 的 processor 配置决定。

## 配置接口

普通推理、统一 pipeline 和 DPO 本地推理的 `infer` 配置均新增：

```json
{
  "image_min_pixels": 65536,
  "image_max_pixels": 589824
}
```

规则：

- 字段类型为 `int | None`，默认 `None`，旧配置行为不变。
- 两个字段必须同时提供或同时省略；显式写两个 `null` 等同于省略。
- 配置值必须是严格正整数；拒绝布尔值、浮点数、字符串、零和负数。
- 必须满足 `image_min_pixels <= image_max_pixels`。
- standalone infer 兼容现有大写旧键 `IMAGE_MIN_PIXELS` / `IMAGE_MAX_PIXELS`；pipeline 和 DPO 使用小写键。
- 当两个字段启用时，预处理 profile 固定为 `llamafactory-0.9.5-qwen-static-v1`。

## 图像处理边界

共享实现放在 `eval_tool/imaging.py`，负责：

- 验证像素边界；
- 按 LLaMAFactory 的顺序执行每一次 resize，而不是只计算最终尺寸后一次缩放；
- 在 resize 完成后转换 RGB；
- 始终返回调用方拥有、可安全关闭的新 PIL 图像；
- 异常时关闭已创建的中间图像。

普通 TSV/base64 推理在 `eval_tool/infer.py::_decode_images` 中调用共享 helper。DPO 在 `eval_tool/dpo_multimodal.py::open_model_messages` 中对已经完成字节哈希验证的原始图像调用同一 helper。这样 resize 均发生在原始颜色模式上，避免当前“先 RGB、后 resize”造成的像素差异。

DPO judge 的 data URL 继续使用未改变的源字节；像素规整只影响本地模型推理图像，不改变资产校验、审计记录或发送给 judge 的原图。

## 推理传播

两个字段需要贯通：

- `InferConfig`、`PipelineInferSettings` 和 pipeline 派生配置；
- 单卡、普通多卡、可续传多卡 worker 的 `QwenVLGenerator` 构造；
- `DpoInferConfig`、spawn DTO、默认 generator factory；
- DPO 顺序推理、递归隔离推理和 worker 推理对 `open_model_messages` 的调用。

`QwenVLGenerator` 保存并防御性验证配置。它的 base64 `generate_batch` 路径在解码入口应用预处理；已经由 DPO 打开的结构化消息不会在 generator 内重复缩放。

## 缓存与可观察性

启用边界时，普通和 DPO 推理 fingerprint 都加入：

- `image_min_pixels`；
- `image_max_pixels`；
- `image_preprocess_profile`。

两个字段均为 `None` 时不改变旧 fingerprint payload，以保留旧配置的续传兼容性。DPO manifest 的 model 段记录两个值和 profile。默认 generator 初始化时打印启用的 LLaMAFactory 兼容边界，以及 checkpoint image processor 暴露的最终 size（若可读取），便于训练/评估审计。

## 测试策略

采用测试驱动开发，覆盖：

- 面积范围内、超过 max、低于 min、`min == max`、28 像素短边、极端宽高比和非 RGB 转换；
- resize 发生在 RGB 转换之前，并按上游顺序执行；
- 普通 infer、pipeline 和 DPO 配置的加载、严格校验和传播；
- 顺序、多卡、可续传 worker 和 DPO DTO 不丢字段；
- 普通与 DPO 图片入口实际向 generator 提供预期尺寸的 RGB 图片，并正确关闭资源；
- 像素设置或算法 profile 改变时 fingerprint 改变，未设置时旧 fingerprint 保持不变；
- DPO manifest 和三个示例配置记录新字段。

## 非目标与限制

- 不把 `image_*` 参数映射成 Hugging Face `min_pixels` / `max_pixels`，避免与训练路径不一致。
- 不改变视频处理。
- 不修改离线选样脚本已有的 JPEG q90 / 最长边 1280 压缩策略；若评估源图经过该脚本而训练源图未经过，仍会存在数据级差异。
- 应用层预处理逻辑可以与 LLaMAFactory 对齐，但逐像素完全一致还要求训练和评估使用相同源图字节、Pillow 版本、Transformers 版本及 checkpoint `preprocessor_config.json`。

## 验收标准

使用示例配置中的 `65536 / 589824` 时，普通评估和 DPO 本地推理都在 RGB 转换前执行 LLaMAFactory 0.9.5 Qwen 静态图像规整；无配置时保持旧行为；所有相关测试和完整测试套件通过，缓存身份能够区分不同视觉预处理设置。
