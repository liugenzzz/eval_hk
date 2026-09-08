#!/usr/bin/env python3
"""校验训练侧写出的信号文件是否符合契约。

纯标准库，不需要安装任何依赖。契约定义见同目录的 训练侧对接说明.md。

    python3 validate_signal.py <signal_dir>/<run_id>/step-600.json
    python3 validate_signal.py --dir <signal_dir>/<run_id>

退出码 0 表示全部通过，1 表示有错误。警告不影响退出码。
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

SCHEMA_VERSION = "1.0"
RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{2,63}$")
EVENTS = ("checkpoint_saved", "run_finished")
FINISH_STATUS = ("completed", "failed", "interrupted")


class Report:
    def __init__(self, label: str) -> None:
        self.label = label
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def ok(self) -> bool:
        return not self.errors

    def render(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        lines = [f"[{mark}] {self.label}"]
        lines += [f"    错误: {item}" for item in self.errors]
        lines += [f"    警告: {item}" for item in self.warnings]
        return "\n".join(lines)


def _require(report: Report, payload: dict, key: str, types: tuple[type, ...]) -> object:
    """取一个必填字段并校验类型；缺失或类型不符时记错误并返回 None。"""
    if key not in payload:
        report.error(f"缺少必填字段 {key}")
        return None
    value = payload[key]
    # bool 是 int 的子类，必须先排掉，否则 is_final=True 能冒充 step。
    if isinstance(value, bool) and bool not in types:
        report.error(f"{key} 类型应为 {types[0].__name__}，实际是 bool")
        return None
    if not isinstance(value, types):
        names = "/".join(t.__name__ for t in types)
        report.error(f"{key} 类型应为 {names}，实际是 {type(value).__name__}")
        return None
    return value


def _check_timestamp(report: Report, value: object, key: str) -> None:
    if not isinstance(value, str):
        return
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        report.error(f"{key} 不是合法的 RFC3339 时间: {value!r}")
        return
    if parsed.tzinfo is None:
        report.error(f"{key} 缺少时区偏移（应形如 2026-09-08T12:00:00+08:00）")


def _check_abs_path(report: Report, value: object, key: str) -> None:
    if not isinstance(value, str):
        return
    path = Path(value)
    if not path.is_absolute():
        report.error(f"{key} 必须是绝对路径: {value!r}")
        return
    # 训练机与评估机是两台机器，路径在本机不存在只是提示，不判错。
    if not path.exists():
        report.warn(f"{key} 在当前机器上不存在（跨机共享盘可忽略）: {value}")


def _check_train_meta(report: Report, payload: dict) -> None:
    meta = _require(report, payload, "train_meta", (dict,))
    if meta is None:
        return
    for key in ("stage", "finetuning_type", "mix_strategy"):
        _require(report, meta, key, (str,))

    strategy = meta.get("mix_strategy")
    if isinstance(strategy, str) and strategy != "concat":
        report.error(
            f"train_meta.mix_strategy 必须是 'concat'，实际是 {strategy!r}。"
            "interleave 下'实际参训条数'不成立，需要先改协议"
        )

    datasets = _require(report, meta, "datasets", (list,))
    if datasets is None:
        return
    if not datasets:
        report.error("train_meta.datasets 不能为空")
        return

    seen: set[str] = set()
    for position, item in enumerate(datasets):
        where = f"train_meta.datasets[{position}]"
        if not isinstance(item, dict):
            report.error(f"{where} 应为对象，实际是 {type(item).__name__}")
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            report.error(f"{where}.name 缺失或不是非空字符串")
        elif name in seen:
            report.error(f"{where}.name 重复: {name!r}")
        else:
            seen.add(name)
        samples = item.get("samples")
        if isinstance(samples, bool) or not isinstance(samples, int):
            report.error(f"{where}.samples 缺失或不是整数")
        elif samples <= 0:
            report.error(f"{where}.samples 必须是正整数，实际是 {samples}")


def _check_common(report: Report, payload: dict) -> tuple[str | None, str | None]:
    version = _require(report, payload, "schema_version", (str,))
    if version is not None and version != SCHEMA_VERSION:
        report.error(f"schema_version 应为 {SCHEMA_VERSION!r}，实际是 {version!r}")

    event = _require(report, payload, "event", (str,))
    if event is not None and event not in EVENTS:
        report.error(f"event 应为 {' 或 '.join(EVENTS)}，实际是 {event!r}")

    run_id = _require(report, payload, "run_id", (str,))
    if isinstance(run_id, str) and not RUN_ID_RE.match(run_id):
        report.error(
            f"run_id 不符合 {RUN_ID_RE.pattern}（小写字母数字下划线，3-64 位）: {run_id!r}"
        )

    _require(report, payload, "request_id", (str,))
    return (event if isinstance(event, str) else None,
            run_id if isinstance(run_id, str) else None)


def _check_checkpoint_saved(report: Report, payload: dict, run_id: str | None) -> None:
    step = _require(report, payload, "step", (int,))
    if isinstance(step, int) and step < 0:
        report.error(f"step 不能为负: {step}")

    epoch = _require(report, payload, "epoch", (int, float))
    if isinstance(epoch, (int, float)) and epoch < 0:
        report.error(f"epoch 不能为负: {epoch}")

    _check_timestamp(report, _require(report, payload, "saved_at", (str,)), "saved_at")
    _check_abs_path(
        report, _require(report, payload, "checkpoint_path", (str,)), "checkpoint_path"
    )
    _check_abs_path(
        report, _require(report, payload, "base_model_path", (str,)), "base_model_path"
    )
    _require(report, payload, "is_final", (bool,))

    request_id = payload.get("request_id")
    if isinstance(request_id, str) and run_id is not None and isinstance(step, int):
        expected = f"{run_id}__step{step}"
        if request_id != expected:
            report.error(
                f"request_id 与 run_id/step 不一致，应为 {expected!r}，实际是 {request_id!r}"
            )

    _check_train_meta(report, payload)


def _check_run_finished(report: Report, payload: dict, run_id: str | None) -> None:
    status = _require(report, payload, "status", (str,))
    if isinstance(status, str) and status not in FINISH_STATUS:
        report.error(f"status 应为 {'/'.join(FINISH_STATUS)}，实际是 {status!r}")

    final_step = _require(report, payload, "final_step", (int,))
    if isinstance(final_step, int) and final_step < 0:
        report.error(f"final_step 不能为负: {final_step}")

    _check_abs_path(
        report,
        _require(report, payload, "final_checkpoint_path", (str,)),
        "final_checkpoint_path",
    )
    _check_timestamp(
        report, _require(report, payload, "finished_at", (str,)), "finished_at"
    )

    request_id = payload.get("request_id")
    if isinstance(request_id, str) and run_id is not None:
        expected = f"{run_id}__finished"
        if request_id != expected:
            report.error(
                f"request_id 应为 {expected!r}，实际是 {request_id!r}"
            )


def validate_file(path: Path, *, require_done: bool = True) -> Report:
    report = Report(str(path))
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        report.error(f"读取失败: {exc}")
        return report
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        report.error(f"JSON 解析失败（第 {exc.lineno} 行第 {exc.colno} 列）: {exc.msg}")
        return report
    if not isinstance(payload, dict):
        report.error(f"顶层应为 JSON 对象，实际是 {type(payload).__name__}")
        return report

    event, run_id = _check_common(report, payload)
    if event == "checkpoint_saved":
        _check_checkpoint_saved(report, payload, run_id)
    elif event == "run_finished":
        _check_run_finished(report, payload, run_id)

    if require_done and not path.with_name(path.name + ".done").exists():
        report.error(
            f"缺少 {path.name}.done。评估侧只扫 .done，没有它这个信号永远不会被处理"
        )

    parent = path.resolve().parent.name
    if run_id is not None and parent and parent != run_id:
        report.warn(
            f"所在目录名 {parent!r} 与 run_id {run_id!r} 不一致"
            "（约定路径是 <signal_dir>/<run_id>/）"
        )
    return report


def validate_dir(directory: Path) -> list[Report]:
    files = sorted(p for p in directory.glob("*.json") if p.suffix == ".json")
    if not files:
        report = Report(str(directory))
        report.error("目录下没有任何 .json 信号文件")
        return [report]

    reports = [validate_file(path) for path in files]

    summary = Report(f"{directory} 目录级检查")
    steps: dict[int, list[str]] = {}
    has_finished = False
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("event") == "run_finished":
            has_finished = True
        step = payload.get("step")
        if isinstance(step, int) and not isinstance(step, bool):
            steps.setdefault(step, []).append(path.name)

    for step, names in sorted(steps.items()):
        if len(names) > 1:
            summary.error(
                f"step {step} 有 {len(names)} 份信号: {', '.join(names)}。"
                "多半是漏了 state.is_world_process_zero 判断"
            )
    if not has_finished:
        summary.warn("没有 finished.json（训练还在跑就正常；已结束的话应当补发）")

    orphans = [
        p.name for p in directory.glob("*.json.done")
        if not p.with_suffix("").exists()
    ]
    for name in orphans:
        summary.error(f"{name} 没有对应的 .json 文件")

    reports.append(summary)
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="校验训练侧信号文件是否符合边训边评契约"
    )
    parser.add_argument("path", help="信号文件路径，或配合 --dir 时的信号目录")
    parser.add_argument(
        "--dir", action="store_true", help="校验整个 <signal_dir>/<run_id> 目录"
    )
    parser.add_argument(
        "--no-done-check", action="store_true", help="跳过 .done 存在性检查"
    )
    args = parser.parse_args(argv)

    target = Path(args.path)
    if args.dir:
        if not target.is_dir():
            print(f"[FAIL] 不是目录: {target}")
            return 1
        reports = validate_dir(target)
    else:
        if not target.is_file():
            print(f"[FAIL] 不是文件: {target}")
            return 1
        reports = [validate_file(target, require_done=not args.no_done_check)]

    for report in reports:
        print(report.render())

    failed = sum(1 for report in reports if not report.ok)
    total = len(reports)
    print(f"\n共 {total} 项检查，{total - failed} 项通过，{failed} 项失败。")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
