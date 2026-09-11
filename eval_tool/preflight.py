"""开跑前的体检：路径、模型、评估集、裁判，一次全查。

存在的意义很实际：整套跑完是几个小时，而最常见的失败原因是一条路径写错。那种错误
要在第 10 秒发现，不是第 3 小时 —— 尤其 ``image_root`` 配错时推理**不会报错**，
模型只是看不见图，几小时之后你拿到一份全是废框的报表。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import PipelineConfig, builder_prompt_root
from .io import truth_path
from .run_eval import _scoring_plan

OK, WARN, BAD = "  ✓", "  !", "  ✗"


class Check:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.errors = 0
        self.warnings = 0

    def ok(self, label: str, detail: Any = "") -> None:
        self.lines.append(f"{OK} {label}{f'  {detail}' if detail else ''}")

    def warn(self, label: str, detail: Any = "") -> None:
        self.warnings += 1
        self.lines.append(f"{WARN} {label}{f'  {detail}' if detail else ''}")

    def bad(self, label: str, detail: Any = "") -> None:
        self.errors += 1
        self.lines.append(f"{BAD} {label}{f'  {detail}' if detail else ''}")

    def path(self, label: str, value: Any, *, required: bool = True, directory: bool = False) -> bool:
        if not value:
            (self.bad if required else self.warn)(label, "（没配）")
            return False
        path = Path(str(value))
        exists = path.is_dir() if directory else path.exists()
        if exists:
            self.ok(label, path)
            return True
        (self.bad if required else self.warn)(label, f"{path}  ← 找不到")
        return False


def preflight(config: PipelineConfig) -> Check:
    check = Check()
    derived = {plan.dataset for plan in config.derive_plans}

    check.lines.append("[评估集]")
    mainline = [key for key in config.enabled_datasets if key not in derived]
    names = sorted({config.datasets[key] for key in mainline})
    for name in names:
        path = truth_path(config.tsv_dir, name)
        if path.is_file():
            rows = sum(1 for _ in path.open(encoding="utf-8")) if path.suffix == ".jsonl" else "?"
            check.ok(f"{name}", f"{path}  {rows} 条")
        else:
            # truth_path 两个都没有时回落到 .tsv，但目标检测读的是 jsonl，
            # 报那个才对得上用户在找的东西。
            check.bad(
                f"{name}",
                f"{config.tsv_dir / (name + '.jsonl')}  ← 找不到。"
                f"文件名由 eval_set 决定，现在要的是 {name}.jsonl",
            )
    for key in derived:
        if key in config.enabled_datasets:
            check.ok(f"{key}", "（派生集，跑的时候自己造）")

    check.lines.append("")
    check.lines.append("[数据路径]")
    seen: dict[str, Any] = {}
    for key in config.enabled_datasets:
        for field in ("image_root", "classes_yaml", "labels_dir", "describe_prompt_dir"):
            value = (config.dataset_params.get(key) or {}).get(field)
            if value:
                seen.setdefault(field, value)
    check.path("image_root", seen.get("image_root"), directory=True)
    check.path("classes_yaml", seen.get("classes_yaml"))
    check.path("labels_dir", seen.get("labels_dir"), required=False, directory=True)
    check.path("describe_prompt_dir", seen.get("describe_prompt_dir"), required=False, directory=True)

    check.lines.append("")
    check.lines.append("[模型]")
    for model in config.models:
        if model.model_path is None:
            check.ok(model.name, "（用现成的预测文件，不推理）")
        else:
            check.path(model.name, model.model_path, directory=True)
    if config.baseline_model not in {model.name for model in config.models}:
        check.bad("baseline_model", f"{config.baseline_model} 不在 models 里")
    else:
        check.ok("baseline_model", config.baseline_model)

    if config.derive_plans:
        check.lines.append("")
        check.lines.append("[派生集]")
        root = builder_prompt_root(seen)
        for plan in config.derive_plans:
            if plan.dataset not in config.enabled_datasets:
                continue
            if not plan.pools:
                check.ok(plan.dataset, f"{plan.mode}  ← 用 {plan.from_model} 的预测")
                continue
            for name, pool in plan.pools.items():
                if pool.is_file():
                    check.ok(f"{plan.dataset} / {name}", pool)
                else:
                    check.bad(
                        f"{plan.dataset} / {name}",
                        f"{pool}  ← 找不到问法池。"
                        + (f"默认按 {root} 下的目录找，" if root else "")
                        + "路径不对就在 derive[].pools 里直接写绝对路径",
                    )

    check.lines.append("")
    check.lines.append("[裁判]")
    check.ok("主裁判", f"{config.judge.model} @ {config.judge.api_base}  指纹 {config.judge.fingerprint}")
    for name, settings in config.cross_check_judges:
        check.ok(f"交叉裁判 {name}", f"{settings.model} @ {settings.api_base}  指纹 {settings.fingerprint}")
    if not config.do_pointwise:
        check.warn("do_pointwise=false", "不调裁判，D 组只出三个代码指标")

    check.lines.append("")
    check.lines.append("[打分器]")
    try:
        plan = _scoring_plan(config.to_eval_config())
        check.ok(f"{len(plan)} 个数据集的 kind 都解析到了打分器")
    except Exception as exc:  # noqa: BLE001 - 原样带给用户，报错里有可选值
        check.bad("kind 解析失败", f"{type(exc).__name__}: {exc}")

    sample = next(
        ((p.get("sample_n"), p.get("sample_seed")) for p in config.dataset_params.values()
         if p.get("sample_n")), None
    )
    check.lines.append("")
    check.lines.append("[其他]")
    if sample:
        check.ok("抽样", f"每份评估集抽 {sample[0]} 条原始样本，种子 {sample[1]}")
    else:
        check.ok("抽样", "关（跑全量）")
    check.ok("输出目录", config.out_dir)
    check.ok("缓存目录", f"{config.cache_dir}  ← 别删，重跑和加模型全靠它")
    return check


def render(check: Check) -> str:
    body = "\n".join(check.lines)
    if check.errors:
        tail = f"\n{check.errors} 处必须先修，{check.warnings} 处提醒。修完再跑。"
    elif check.warnings:
        tail = f"\n没有阻塞问题，{check.warnings} 处提醒 —— 确认一下是不是有意为之。"
    else:
        tail = "\n全部通过，可以跑了。"
    return f"{body}\n{tail}"
