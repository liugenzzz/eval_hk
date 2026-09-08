"""坐标空间（scale / origin）从数据里读，不写死 1000。

构建端把 ``bbox_scale`` / ``coordinate_mode`` 逐样本写进 metadata。打分器和合规层
共用这一份解析，两边各写一遍迟早对不上。
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

DEFAULT_SCALE = 1000
DEFAULT_ORIGIN = 0


def scale_of(data: pd.DataFrame, params: Mapping[str, Any]) -> tuple[int, int]:
    """坐标空间从数据里读，不写死 1000。

    构建端把 ``bbox_scale`` / ``coordinate_mode`` 逐样本写进 metadata。全批必须一致，
    否则 5% 的换算会静默算错 —— 一半样本按 1000 算、一半按别的算，平均出来的数
    没有意义。所以这里发现不一致直接报错。
    """
    scale = int(params.get("scale") or DEFAULT_SCALE)
    origin = int(params.get("origin") or DEFAULT_ORIGIN)
    for column in ("meta.bbox_scale", "bbox_scale"):
        if column not in data.columns:
            continue
        values = {int(v) for v in pd.Series(data[column]).dropna().tolist()}
        if not values:
            continue
        if len(values) > 1:
            raise ValueError(f"{column} 在同一批数据里有多个取值：{sorted(values)}，坐标空间必须一致")
        found = values.pop()
        if params.get("scale") and int(params["scale"]) != found:
            raise ValueError(f"params.scale={params['scale']} 与数据里的 {column}={found} 不一致")
        scale = found
        break
    return scale, origin
