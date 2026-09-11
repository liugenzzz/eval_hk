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
    challengers = [m.name for m in config.models if m.name != config.baseline_model]

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
    check.lines.append("[各切片的样本数]")
    _slice_sizes(check, config, mainline)

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
        from_models = {plan.from_model for plan in config.derive_plans if plan.from_model}
        if len(challengers) > 1 and from_models:
            # 派生集是拿某一个模型的输出造的，然后所有模型都在这一份上评。对
            # reverse_consistency 没问题（问的是固定的一批框），但链路衰减率
            # (chain_decay) 的定义是「模型接着**自己的**历史往下答」，拿 A 的历史
            # 喂给 B 测出来的不是 B 的链路。
            check.warn(
                "derive_from",
                f"{'/'.join(sorted(from_models))} —— 派生集只由它一个造，"
                "chain_decay 只对它成立；别的 checkpoint 那几行是「接着它的历史答」，"
                "不是各自的链路",
            )
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
    if config.do_pairwise and len(challengers) > 2:
        # 成对判定是「每个挑战者 × 基线 × 每条样本 × 正反两个方向」。评一串 checkpoint
        # 时这个数是乘出来的，而 checkpoint 之间的排序 pointwise 的绝对分就够看了。
        check.warn(
            "do_pairwise=true",
            f"有 {len(challengers)} 个非基线模型，成对判定的裁判调用是它们乘出来的。"
            "只是想给 checkpoint 排序的话，设成 false 更划算",
        )

    check.lines.append("")
    check.lines.append("[推理]")
    if config.infer.gpu_ids:
        workers = len(config.infer.gpu_ids) * config.infer.workers_per_gpu
        check.ok(
            "数据并行",
            f"GPU {config.infer.gpu_ids} × 每卡 {config.infer.workers_per_gpu} worker "
            f"= {workers} 个进程，各加载一份模型，batch_size={config.infer.batch_size}",
        )
        if config.infer.device_map and config.infer.device_map != "auto":
            check.warn("device_map", "配了 gpu_ids 时这个字段用不上（每个 worker 独占一张卡）")
        check.warn(
            "gpu_ids",
            "写的是**物理**卡号，每个 worker 会自己设 CUDA_VISIBLE_DEVICES。"
            "命令行上就别再设 CUDA_VISIBLE_DEVICES 了，两边会打架",
        )
    else:
        check.ok("单进程推理", f"device_map={config.infer.device_map}，batch_size={config.infer.batch_size}"
                              "（模型单卡放得下的话，配 gpu_ids 做数据并行快得多）")

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


def _slice_sizes(check: Check, config: PipelineConfig, mainline: list[str]) -> None:
    """每个切片实际有多少条。

    十一个数据集读的是同一份 test.jsonl，各自 select 一个子集，再叠上抽样。跑起来
    只看到「共 593 条」的时候没人算得出这 593 是怎么来的 —— 在这里一次全列出来，
    跑之前就知道每一档有多少样本、够不够下结论（n < 30 报表上不给百分比）。
    """
    from .io import load_truth_dataset

    for key in mainline:
        try:
            # 不读图：这里只数行数，为了几个数字把整批图编成 base64 没有道理
            rows = len(load_truth_dataset(
                config.tsv_dir, config.datasets[key], config.dataset_params.get(key) or {},
                need_images=False,
            ))
        except Exception as exc:  # noqa: BLE001 - 体检不该因为一个切片读不了就中断
            check.bad(key, f"{type(exc).__name__}: {exc}")
            continue
        if not rows:
            # 数据里确实没有这种任务时它就是 0，不该拦住整趟；但也不能不说 ——
            # 这一格在报表上会整个缺席。
            check.warn(key, "0 条  ← 这个数据集不会被评估（数据里没有这种任务？）")
        elif rows < 30:
            check.warn(key, f"{rows} 条  ← n < 30，报表上这一档不给百分比")
        else:
            check.ok(key, f"{rows} 条")


def render(check: Check) -> str:
    body = "\n".join(check.lines)
    if check.errors:
        tail = f"\n{check.errors} 处必须先修，{check.warnings} 处提醒。修完再跑。"
    elif check.warnings:
        tail = f"\n没有阻塞问题，{check.warnings} 处提醒 —— 确认一下是不是有意为之。"
    else:
        tail = "\n全部通过，可以跑了。"
    return f"{body}\n{tail}"
