"""每张图的完整类别集合 —— CHAIR 幻觉率的分母。

CHAIR 要回答「模型提到的类别，图里到底有没有」。这需要一份**完整**的类别清单：
漏掉一个类，模型提到它就会被冤枉成幻觉。

三个来源，按可信度排：

1. ``metadata.image_classes`` —— 构建端直接给的全图类别集合。目前没有这个字段，
   将来加了就自动优先用它。
2. **原始 YOLO 标注文件**（``params.labels_dir`` + ``metadata.source_annotation``）
   —— 每条样本都带着它是从哪个标注文件来的，那个文件里有全图每个框的 class_id，
   配上 classes.yaml 就能还原出完整集合。**这是当前推荐的路子：覆盖 100%，
   数据侧不用改任何东西。**
3. ``metadata.inventory`` —— 只有 ``inventory_locate`` 那一种样本带，覆盖不到
   四分之一的描述样本。

用**原始标注**而不是构建期过滤后的清单，是有意的：构建端会按 ``min_area_ratio`` /
``min_short_side_px`` 滤掉太小太糊的框，那些目标**图里是真存在的**。拿过滤后的集合算
CHAIR，模型提到一个小卡车就会被记成编造 —— 那是冤枉。宁可放过几个「说了个看不清的
东西」，也不要把说对了的记成幻觉。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .classes import ClassTable


class ImageClassIndex:
    """按需读标注文件，缓存结果。同一张图上有十几条样本，读一次就够。"""

    def __init__(self, labels_dir: str | Path | None, table: ClassTable | None):
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.table = table
        self._cache: dict[str, tuple[str, ...] | None] = {}

    @property
    def enabled(self) -> bool:
        return self.labels_dir is not None and self.table is not None

    def classes_of(self, annotation_name: object) -> tuple[str, ...] | None:
        """读一个 YOLO 标注文件，返回图里出现过的全部类别名。

        文件不存在、读不动、或者出现了类别表里没有的 class_id 时返回 None ——
        那说明这份标注和这份类别表对不上，给一个残缺的集合去算 CHAIR 比不算更糟。
        """
        if not self.enabled:
            return None
        name = str(annotation_name or "").strip()
        if not name:
            return None
        if name in self._cache:
            return self._cache[name]
        self._cache[name] = self._read(name)
        return self._cache[name]

    def _read(self, name: str) -> tuple[str, ...] | None:
        path = self.labels_dir / name  # type: ignore[operator]
        if not path.exists():
            return None
        names: set[str] = set()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return None
        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                return None
            label = self.table.id2name.get(class_id)  # type: ignore[union-attr]
            if label is None:
                # 标注里的 class_id 在类别表里查不到：两份文件对不上，
                # 硬算出来的集合是残缺的。
                return None
            names.add(str(label).strip())
        return tuple(sorted(names))


def declared_image_classes(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    """构建端将来直接给出 ``metadata.image_classes`` 时走这条。"""
    for column in ("meta.image_classes", "image_classes"):
        value = row.get(column)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            names = tuple(str(v).strip() for v in value if str(v).strip())
            return names or None
        text = str(value).strip()
        if text:
            return tuple(part.strip() for part in text.split(",") if part.strip())
    return None
