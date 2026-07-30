"""端到端冒烟检查：JSON -> TSV -> 推理 -> 评分 -> 报告，全程无需 GPU 和裁判 API。

用假模型（固定文本输出）和假裁判（固定判对/判 A 胜）替换真实依赖，
验证整条通路的数据格式兼容性。任何一步炸了都会带着完整 traceback 退出。

用法：python e2e_check.py
"""

from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
WORK = ROOT / "e2e_tmp"


def step(msg: str) -> None:
    print(f"\n===== {msg} =====", flush=True)


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    try:
        from PIL import Image

        Image.new("RGB", (32, 32), color).save(path, format="JPEG")
    except ImportError:
        path.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9")


def build_sample_json(data_dir: Path) -> Path:
    img_dir = data_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img1, img2, img3 = img_dir / "1.jpg", img_dir / "2.jpg", img_dir / "3.jpg"
    make_image(img1, (200, 30, 30))
    make_image(img2, (30, 200, 30))
    make_image(img3, (30, 30, 200))
    records = [
        {  # 单轮 + 单图
            "id": "vqa_000001",
            "image": [str(img1)],
            "conversations": [
                {"from": "human", "value": "<image>\n图中这艘白色大型舰船整体是做什么用的？"},
                {"from": "gpt", "value": "图中是636A型中国海道测量船，主要执行水文测量任务。"},
            ],
        },
        {  # 多轮 + 多图（第二轮引入第二张图）
            "id": "vqa_000002",
            "image": [str(img2), str(img3)],
            "conversations": [
                {"from": "human", "value": "<image>\n图中这架直升机是什么型号？"},
                {"from": "gpt", "value": "美国CH-47F Block II支奴干通用运输直升机。"},
                {"from": "human", "value": "<image>\n这张图里它的旋翼有什么特点？"},
                {"from": "gpt", "value": "双旋翼纵列布局，后掠尖端转子叶片，可额外提供1500磅升力。"},
            ],
        },
        {  # 单轮 + 多图
            "id": "vqa_000003",
            "image": [str(img1), str(img2)],
            "conversations": [
                {"from": "human", "value": "<image>\n<image>\n对比两图中的装备类型。"},
                {"from": "gpt", "value": "第一张为海道测量船，第二张为运输直升机。"},
            ],
        },
    ]
    src = data_dir / "sample_sharegpt.json"
    src.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"样例数据: {len(records)} 条记录 -> {src}")
    return src


class FakeGenerator:
    """替代 QwenVLGenerator：记录收到的 prompt 结构，返回可区分的固定回答。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.seen: list[object] = []

    def generate_batch(self, prompts, image_b64s=None):
        out = []
        for prompt in prompts:
            self.seen.append(prompt)
            if isinstance(prompt, list):  # 多轮 chat 结构
                n_hist = sum(1 for t in prompt if t.get("role") == "assistant")
                out.append(f"[{self.model_name}] 多轮回答（携带{n_hist}轮历史）")
            else:
                out.append(f"[{self.model_name}] 单轮回答：这是对问题的描述。")
        return out


def fake_judge_post(self, system_prompt: str, user_text: str, image_b64=None, mime="image/jpeg") -> str:
    if "回答A" in user_text:  # pairwise
        return '{"winner": "A", "reason": "e2e stub"}'
    return '{"correct": true, "reason": "e2e stub"}'


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir()

    step("1/5 生成样例 ShareGPT JSON")
    src = build_sample_json(WORK / "data")

    step("2/5 JSON -> aero_vqa.tsv (convert_vqa_json)")
    from eval_tool.convert_vqa_json import convert

    tsv_dir = WORK / "LMUData"
    convert(src, tsv_dir)
    tsv_text = (tsv_dir / "aero_vqa.tsv").read_text(encoding="utf-8-sig")
    n_rows = len(tsv_text.strip().splitlines()) - 1
    print(f"TSV 行数: {n_rows} (应为 4：单轮3行 + 多轮1行)")
    assert n_rows == 4, f"TSV 行数不对: {n_rows}"

    step("3/5 推理 (run_infer + 假模型)")
    from eval_tool.config import InferConfig
    from eval_tool.run_infer import run as run_infer

    pred_paths = {}
    for model_name in ("base", "sft"):
        cfg = InferConfig(
            model_name=model_name,
            model_path=Path("unused"),
            tsv_dir=tsv_dir,
            out_dir=WORK / "work_dir" / model_name,
            datasets={"vqa": "aero_vqa"},
            prompt_files={"vqa": ROOT / "prompts" / "infer_vqa_raw.txt"},
        )
        gen = FakeGenerator(model_name)
        written = run_infer(cfg, generator=gen)
        pred_paths[model_name] = written["vqa"]
        multi = [p for p in gen.seen if isinstance(p, list)]
        print(f"{model_name}: 生成 {len(gen.seen)} 条，其中多轮 chat 结构 {len(multi)} 条 -> {written['vqa']}")
        assert len(multi) == 1, "应有 1 条多轮样本走 chat 结构"
        assert [t["role"] for t in multi[0]] == ["user", "assistant", "user"], "多轮消息应为 user/assistant/user"

    step("4/5 评分 (run_eval + 假裁判)")
    from eval_tool import judge as judge_mod

    judge_mod.JudgeClient._post = fake_judge_post  # 拦截真实 API 调用
    from eval_tool.config import EvalConfig, ModelConfig
    from eval_tool.run_eval import run as run_eval

    eval_cfg = EvalConfig(
        tsv_dir=tsv_dir,
        out_dir=WORK / "eval_report",
        cache_dir=WORK / "eval_cache",
        datasets={"vqa": "aero_vqa"},
        models=[
            ModelConfig(name="base", paths={"vqa": str(pred_paths["base"])}),
            ModelConfig(name="sft", paths={"vqa": str(pred_paths["sft"])}),
        ],
        baseline_model="base",
        max_workers=4,
        enabled_datasets=["vqa"],
    )
    written = run_eval(eval_cfg)
    for name in sorted(written):
        print(f"  产出: {name}")

    step("5/5 校验报告内容")
    import pandas as pd

    score = pd.read_csv(written["score_summary.csv"])
    print(score.to_string(index=False))
    assert "total_score" in score.columns, "score_summary 缺 total_score"
    assert "单轮" in score.columns and "多轮" in score.columns, "score_summary 应有 单轮/多轮 分列"
    assert len(score) == 2, "应有 base/sft 两行"
    pairwise = pd.read_csv(written["pairwise_vs_baseline.csv"])
    cats = set(pairwise.get("category", pd.Series(dtype=str)).dropna())
    print(f"pairwise 覆盖类别: {cats}")

    combined = pd.read_excel(written["judge_detail_all.xlsx"])
    models_in_detail = set(combined["model"])
    print(f"judge_detail_all.xlsx: {len(combined)} 行，模型: {models_in_detail}")
    assert models_in_detail == {"base", "sft"}, "裁判明细合并文件应同时含两个模型"
    assert "judge_reason" in combined.columns, "裁判明细应含 judge_reason 列"

    print("\n✅ 端到端通路 OK：转换 -> 推理(单轮/多轮/多图) -> 评分 -> 单轮/多轮分列报告")
    return 0


if __name__ == "__main__":
    sys.exit(main())
