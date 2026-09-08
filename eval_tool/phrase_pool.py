"""问法池：读构建端 ``prompts/<task>/*.txt`` 的模板池，按占位符渲染。

问法扰动（§11.2）测的是对问法的过拟合。SFT 数据的问法有限，模型很容易学成「看到某个
固定句式才输出坐标」，换个说法就崩 —— 这个失败模式在真实使用中极常见，而标准评测集
完全测不出来（它们的问法也是固定的）。

池子文件的格式（与构建端一致）：``#`` 开头是注释，``#!`` 开头是构建期的校验指令，
其余每行一个模板，占位符形如 ``{label}`` / ``{attribute}`` / ``{bbox}`` / ``{mw}``。
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

_PLACEHOLDER = re.compile(r"\{([a-zA-Z_]+)\}")


@dataclass(frozen=True)
class PhrasePool:
    name: str
    templates: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "PhrasePool":
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"找不到问法池文件：{file}")
        templates = tuple(
            line.strip()
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
        if not templates:
            raise ValueError(f"{file} 里没有模板行")
        return cls(name=file.stem, templates=templates)

    def usable(self, values: Mapping[str, str]) -> tuple[str, ...]:
        """只保留占位符能被填满的模板。

        ``{mw}``（量词）没给就跳过那几条，不硬填一个「个」—— 量词用错会让问句读起来
        不像人话，而我们测的是模型对**正常问法**的稳定性，不是对病句的容忍度。
        """
        available = {key for key, value in values.items() if str(value or "").strip()}
        return tuple(
            template
            for template in self.templates
            if set(_PLACEHOLDER.findall(template)) <= available
        )

    def render(
        self, values: Mapping[str, str], count: int, seed_key: str, exclude: Iterable[str] = ()
    ) -> tuple[str, ...]:
        """挑 ``count`` 条互不相同的问法并渲染。

        选择由 ``seed_key``（样本 id）定，同一条样本每次跑都拿到同一批问法 ——
        问法本身变了，扰动实验前后就不可比了。
        """
        candidates = [t for t in self.usable(values) if _render(t, values) not in set(exclude)]
        if not candidates:
            return ()
        digest = hashlib.sha256(f"{self.name}|{seed_key}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        # sorted() 让选择只取决于 seed_key，不取决于文件里模板的先后 —— 池子里插一行
        # 注释都不该改变已经跑过的样本抽到哪几句。
        ordered = sorted(candidates)
        picked = ordered if len(ordered) <= count else rng.sample(ordered, count)
        return tuple(_render(template, values) for template in picked)


def _render(template: str, values: Mapping[str, str]) -> str:
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value or ""))
    return out


def load_pools(paths: Mapping[str, str | Path]) -> dict[str, PhrasePool]:
    return {str(key): PhrasePool.load(path) for key, path in paths.items()}


def format_bbox(box: Sequence[float]) -> str:
    """构建端问句里的坐标写法：``[x1,y1,x2,y2]``，无空格、整数。"""
    return "[" + ",".join(str(int(round(float(v)))) for v in box) + "]"
