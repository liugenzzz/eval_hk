"""多框任务的预测-真值配对：以 IoU 为收益的匈牙利匹配（全局最优一对一）。

为什么不贪心：贪心按 IoU 从高到低抢配对，两个挨得近的目标会被同一个预测框
先抢走一个，剩下的配不上，于是同时多记一次漏检和一次误检 —— 报表上看起来是
模型漏了，实际是匹配算法挑错了。匈牙利求的是全局最优，同一批输入永远给出
同一个配对结果。

自己实现而不是引 scipy：代码打分器要求「无网络、可复现、逐位相同」，评估机在
内网离线装依赖，为一个函数拖进 scipy 不划算。框数是几个到几十个，O(n^3) 的
代价可以忽略。

实现是 Jonker-Volgenant 风格的 O(n^3) 匈牙利（最小代价完全匹配），代价矩阵取
``1 - IoU``；配不上（IoU 低于阈值）的配对在最后一步剔掉。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .bbox import Box, iou


@dataclass(frozen=True)
class Matching:
    pairs: tuple[tuple[int, int], ...]   # (gt 下标, pred 下标)，按 gt 下标升序
    missed: tuple[int, ...]              # 漏检：gt 有、pred 没有
    spurious: tuple[int, ...]            # 误检：pred 有、gt 没有


def solve_min_cost(cost: Sequence[Sequence[float]]) -> list[int]:
    """最小代价完全匹配。返回 assignment[i] = 分配给行 i 的列下标（-1 表示没有）。

    行数可以少于列数；行多于列时多出来的行拿不到列。
    """
    n_rows = len(cost)
    n_cols = len(cost[0]) if n_rows else 0
    if not n_rows or not n_cols:
        return [-1] * n_rows
    # 内部按「行 <= 列」处理，反过来时转置再把结果翻回去。
    if n_rows > n_cols:
        transposed = [[cost[r][c] for r in range(n_rows)] for c in range(n_cols)]
        col_for_row = [-1] * n_rows
        for col, row in enumerate(solve_min_cost(transposed)):
            if row >= 0:
                col_for_row[row] = col
        return col_for_row

    INF = math.inf
    # u/v 是对偶变量，way 记录交替路径。下标从 1 开始，0 号是虚拟行/列。
    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)
    p = [0] * (n_cols + 1)       # p[col] = 匹配到该列的行
    way = [0] * (n_cols + 1)
    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n_cols + 1)
        used = [False] * (n_cols + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n_cols + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = [-1] * n_rows
    for j in range(1, n_cols + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    return assignment


def match_boxes(gt: Sequence[Box], pred: Sequence[Box], iou_gate: float = 0.5) -> Matching:
    """按 IoU 做一对一匹配，IoU < iou_gate 的配对不算数。"""
    if not gt or not pred:
        return Matching(pairs=(), missed=tuple(range(len(gt))), spurious=tuple(range(len(pred))))

    cost = [[1.0 - iou(g, p) for p in pred] for g in gt]
    assignment = solve_min_cost(cost)

    pairs: list[tuple[int, int]] = []
    matched_pred: set[int] = set()
    missed: list[int] = []
    for gt_index, pred_index in enumerate(assignment):
        if pred_index >= 0 and iou(gt[gt_index], pred[pred_index]) >= iou_gate:
            pairs.append((gt_index, pred_index))
            matched_pred.add(pred_index)
        else:
            missed.append(gt_index)
    spurious = [i for i in range(len(pred)) if i not in matched_pred]
    return Matching(pairs=tuple(pairs), missed=tuple(missed), spurious=tuple(spurious))
