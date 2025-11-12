# -*- coding: utf-8 -*-
"""
Search Strategies Lab (Extended Heuristics & Weighted A*)
---------------------------------------------------------
实现并比较以下搜索策略（4-邻接、单位步长）：
- BFS（宽度优先）
- DFS（深度优先）
- UCS（等代价搜索 / Dijkstra）
- Greedy Best-First（贪心，Manhattan 启发）
- A*（三种启发：Manhattan / Chebyshev / Octile）
- Weighted A*：f(n) = g(n) + w * h(n), 其中 w >= 1（示例 w=1.5, 2.0）

输出指标：
- Path Length (steps)  最终路径步数（若无路径则为 0）
- Nodes Expanded       展开结点数（可视作“搜索区域面积”，= 被真正弹出/处理的一次性结点计数）
- Max Frontier Size    搜索过程中 frontier（队列/堆/栈）尺寸峰值（反映峰值内存/边界）

额外导出：
- /<out_dir>/Astar_path_overlay.txt  （A* Manhattan 的叠加文本，按题目要求）
- /<out_dir>/Astar_Manhattan_overlay.txt
- /<out_dir>/Astar_Chebyshev_overlay.txt
- /<out_dir>/Astar_Octile_overlay.txt
- /<out_dir>/Weighted_Astar_w=1.5_Manhattan_overlay.txt
- /<out_dir>/Weighted_Astar_w=2.0_Manhattan_overlay.txt
- /<out_dir>/search_metrics_extended.csv

地图符号（输入）：
  '#' = 墙/障碍
  '.' = 空地
  '$' = 起点
  '@' = 终点

叠加符号（输出）：
  '!' = 已展开区域（expanded）
  '#' = 最终路径（注意：与墙同符号，按题意要求）
  '$' = 起点
  '@' = 终点
  其余保持原样

用法（可选）：
  python search_lab.py --map /mnt/data/maze-grid.txt --w 1.5 2.0
"""

from __future__ import annotations
import argparse
import csv
import heapq
import math
import os
from collections import deque
from typing import Dict, Tuple, List, Optional, Set

Coord = Tuple[int, int]


# =========================
# 基础：读图与网格工具
# =========================
def load_grid(path: str):
    """
    读取 ASCII 栅格地图，返回：
      grid  : List[List[str]] 字符矩阵
      start : (r,c) 起点
      goal  : (r,c) 终点
    """
    with open(path, "r", encoding="utf-8") as f:
        # 去掉空行，rstrip 去除末尾换行
        lines = [line.rstrip("\n") for line in f if line.strip()]
    grid = [list(row) for row in lines]

    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == "$":
                start = (r, c)
            elif ch == "@":
                goal = (r, c)

    if start is None or goal is None:
        raise ValueError("地图必须包含一个起点 '$' 与一个终点 '@'。")
    return grid, start, goal


def in_bounds(grid: List[List[str]], r: int, c: int) -> bool:
    return 0 <= r < len(grid) and 0 <= c < len(grid[0])


def is_free(ch: str) -> bool:
    """可通行：非墙('#')均可，包括 '$' 与 '@'。"""
    return ch != "#"


def neighbors4(grid: List[List[str]], rc: Coord) -> List[Coord]:
    """
    4-邻接（上/下/左/右）。若需 8-邻接，可自行扩展。
    """
    r, c = rc
    res: List[Coord] = []
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if in_bounds(grid, nr, nc) and is_free(grid[nr][nc]):
            res.append((nr, nc))
    return res


# =========================
# 启发式函数（Heuristics）
# =========================
def h_manhattan(a: Coord, b: Coord) -> float:
    """
    曼哈顿距离（L1）：|dr| + |dc|
    - 4-邻接 + 单位步长 下可采纳（admissible）且一致（consistent）
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def h_chebyshev(a: Coord, b: Coord) -> float:
    """
    切比雪夫：max(|dr|, |dc|)
    - 8-邻接常用；在 4-邻接下也不超过曼哈顿，仍可采纳但更保守
    """
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return max(dx, dy)


def h_octile(a: Coord, b: Coord) -> float:
    """
    Octile 距离（8-邻接中对角代价为 sqrt(2) 的常见启发）：
      dx + dy + (sqrt(2)-2)*min(dx, dy)
    在 4-邻接下 ≤ 曼哈顿，亦可采纳但更保守。
    """
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    return dx + dy + (math.sqrt(2) - 2.0) * min(dx, dy)


# =========================
# 路径重建（通用）
# =========================
def reconstruct_path(
    came_from: Dict[Coord, Optional[Coord]], start: Coord, goal: Coord
) -> List[Coord]:
    """
    根据父指针字典 came_from 重建 start->goal 的路径。
    若 goal 不在 came_from，返回空列表（表示不可达）。
    """
    if goal not in came_from:
        return []
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from[cur]
        if cur is None:
            break  # 到达 start（其父为 None）
        path.append(cur)
    path.reverse()
    return path


# =========================
# 各搜索算法实现
# =========================
def bfs(grid: List[List[str]], start: Coord, goal: Coord):
    """
    BFS：队列（FIFO）。在单位步长无权图上，可找到最少步数路径。
    指标统计：
      - nodes_expanded：每次从队列弹出的唯一结点计数
      - max_frontier  ：队列长度峰值
    """
    frontier = deque([start])
    came_from: Dict[Coord, Optional[Coord]] = {start: None}
    expanded: Set[Coord] = set()
    max_frontier, nodes_expanded = 1, 0

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        current = frontier.popleft()
        if current in expanded:
            continue
        expanded.add(current)
        nodes_expanded += 1

        if current == goal:
            break

        for nxt in neighbors4(grid, current):
            if nxt not in came_from:  # 未访问
                came_from[nxt] = current
                frontier.append(nxt)

    path = reconstruct_path(came_from, start, goal)
    return {
        "algorithm": "BFS",
        "path": path,
        "path_length": max(0, len(path) - 1),
        "nodes_expanded": nodes_expanded,
        "max_frontier": max_frontier,
        "expanded_set": expanded,
    }


def dfs(grid: List[List[str]], start: Coord, goal: Coord):
    """
    DFS：栈（LIFO）。不保证最短，但能找到一条路径（若存在）。
    """
    stack = [start]
    came_from: Dict[Coord, Optional[Coord]] = {start: None}
    expanded: Set[Coord] = set()
    max_frontier, nodes_expanded = 1, 0

    while stack:
        max_frontier = max(max_frontier, len(stack))
        current = stack.pop()
        if current in expanded:
            continue
        expanded.add(current)
        nodes_expanded += 1

        if current == goal:
            break

        for nxt in neighbors4(grid, current):
            if nxt not in came_from:
                came_from[nxt] = current
                stack.append(nxt)

    path = reconstruct_path(came_from, start, goal)
    return {
        "algorithm": "DFS",
        "path": path,
        "path_length": max(0, len(path) - 1),
        "nodes_expanded": nodes_expanded,
        "max_frontier": max_frontier,
        "expanded_set": expanded,
    }


def ucs(grid: List[List[str]], start: Coord, goal: Coord):
    """
    UCS（Uniform Cost Search / Dijkstra）：优先队列按 g(n)（到达代价）排序。
    在单位步长下与 BFS 的最优性一致。
    """
    counter = 0  # 打破堆的平手，保证稳定性
    frontier: List[Tuple[float, int, Coord]] = [(0.0, counter, start)]
    came_from: Dict[Coord, Optional[Coord]] = {start: None}
    g_cost: Dict[Coord, float] = {start: 0.0}
    expanded: Set[Coord] = set()
    max_frontier, nodes_expanded = 1, 0

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        g, _, current = heapq.heappop(frontier)
        if current in expanded:
            continue
        expanded.add(current)
        nodes_expanded += 1

        if current == goal:
            break

        for nxt in neighbors4(grid, current):
            new_g = g + 1.0
            if nxt not in g_cost or new_g < g_cost[nxt]:
                g_cost[nxt] = new_g
                came_from[nxt] = current
                counter += 1
                heapq.heappush(frontier, (new_g, counter, nxt))

    path = reconstruct_path(came_from, start, goal)
    return {
        "algorithm": "UCS",
        "path": path,
        "path_length": max(0, len(path) - 1),
        "nodes_expanded": nodes_expanded,
        "max_frontier": max_frontier,
        "expanded_set": expanded,
    }


def greedy_best_first(
    grid: List[List[str]],
    start: Coord,
    goal: Coord,
    heuristic=h_manhattan,
    label: str = "Greedy (Manhattan)",
):
    """
    贪心最佳优先：优先级仅为 h(n)，忽略 g(n)，不保证最优。
    """
    counter = 0
    frontier: List[Tuple[float, int, Coord]] = [(heuristic(start, goal), counter, start)]
    came_from: Dict[Coord, Optional[Coord]] = {start: None}
    visited: Set[Coord] = {start}
    expanded: Set[Coord] = set()
    max_frontier, nodes_expanded = 1, 0

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        _, _, current = heapq.heappop(frontier)
        if current in expanded:
            continue
        expanded.add(current)
        nodes_expanded += 1

        if current == goal:
            break

        for nxt in neighbors4(grid, current):
            if nxt not in visited:
                visited.add(nxt)
                came_from[nxt] = current
                counter += 1
                heapq.heappush(frontier, (heuristic(nxt, goal), counter, nxt))

    path = reconstruct_path(came_from, start, goal)
    return {
        "algorithm": label,
        "path": path,
        "path_length": max(0, len(path) - 1),
        "nodes_expanded": nodes_expanded,
        "max_frontier": max_frontier,
        "expanded_set": expanded,
    }


def a_star(
    grid: List[List[str]],
    start: Coord,
    goal: Coord,
    heuristic=h_manhattan,
    label: str = "A* (Manhattan)",
):
    """
    A*：f(n) = g(n) + h(n)
    - 若 h 可采纳且一致（Manhattan 在 4-邻接即如此），则在单位步长图上找到最短步数路径。
    """
    counter = 0
    frontier: List[Tuple[float, int, Coord]] = [(heuristic(start, goal), counter, start)]
    came_from: Dict[Coord, Optional[Coord]] = {start: None}
    g_cost: Dict[Coord, float] = {start: 0.0}
    expanded: Set[Coord] = set()
    max_frontier, nodes_expanded = 1, 0

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        f, _, current = heapq.heappop(frontier)
        if current in expanded:
            continue
        expanded.add(current)
        nodes_expanded += 1

        if current == goal:
            break

        for nxt in neighbors4(grid, current):
            new_g = g_cost[current] + 1.0
            if nxt not in g_cost or new_g < g_cost[nxt]:
                g_cost[nxt] = new_g
                came_from[nxt] = current
                counter += 1
                heapq.heappush(frontier, (new_g + heuristic(nxt, goal), counter, nxt))

    path = reconstruct_path(came_from, start, goal)
    return {
        "algorithm": label,
        "path": path,
        "path_length": max(0, len(path) - 1),
        "nodes_expanded": nodes_expanded,
        "max_frontier": max_frontier,
        "expanded_set": expanded,
    }


def weighted_a_star(
    grid: List[List[str]],
    start: Coord,
    goal: Coord,
    heuristic=h_manhattan,
    w: float = 1.5,
    label: Optional[str] = None,
):
    """
    Weighted A*：f(n) = g(n) + w * h(n)，其中 w >= 1
      - w=1 等价标准 A*（保持最优）
      - w>1 往目标更激进，一般扩展更少/更快，但可能非最优
    """
    if w < 1.0:
        raise ValueError("Weighted A* 需满足 w >= 1.0")
    if label is None:
        label = f"Weighted A* w={w:.1f} (Manhattan)" if heuristic == h_manhattan else f"Weighted A* w={w:.1f}"

    counter = 0
    frontier: List[Tuple[float, int, Coord]] = [(w * heuristic(start, goal), counter, start)]
    came_from: Dict[Coord, Optional[Coord]] = {start: None}
    g_cost: Dict[Coord, float] = {start: 0.0}
    expanded: Set[Coord] = set()
    max_frontier, nodes_expanded = 1, 0

    while frontier:
        max_frontier = max(max_frontier, len(frontier))
        f, _, current = heapq.heappop(frontier)
        if current in expanded:
            continue
        expanded.add(current)
        nodes_expanded += 1

        if current == goal:
            break

        for nxt in neighbors4(grid, current):
            new_g = g_cost[current] + 1.0
            # Weighted A* 仍按 g 更新（最短 g），只是入堆优先级为 g + w*h
            if nxt not in g_cost or new_g < g_cost[nxt]:
                g_cost[nxt] = new_g
                came_from[nxt] = current
                counter += 1
                heapq.heappush(frontier, (new_g + w * heuristic(nxt, goal), counter, nxt))

    path = reconstruct_path(came_from, start, goal)
    return {
        "algorithm": label,
        "path": path,
        "path_length": max(0, len(path) - 1),
        "nodes_expanded": nodes_expanded,
        "max_frontier": max_frontier,
        "expanded_set": expanded,
    }


# =========================
# 叠加文本（标注展开与路径）
# =========================
def build_overlay_text(
    grid: List[List[str]], expanded_set: Set[Coord], path: List[Coord], title: str
) -> str:
    """
    构造叠加文本：
      - 展开过的 '.' 标记为 '!'
      - 最终路径标记为 '#'
      - '$' 与 '@' 保持，墙保持 '#'
    注意：题目要求路径也用 '#', 与墙相同。为避免混淆，文件头部附带图例。
    """
    g = [row[:] for row in grid]  # 深拷贝
    # 标记展开
    for (r, c) in expanded_set:
        if g[r][c] == ".":
            g[r][c] = "!"
    # 标记路径（避免覆盖起点/终点）
    path_set = set(path)
    for (r, c) in path_set:
        if g[r][c] not in ("$", "@"):
            g[r][c] = "^"

    header = [
        title,
        "Legend: '!'=expanded, '#'=final path, '$'=start, '@'=goal, walls='#'",
        "-" * 80,
    ]
    body = ["".join(row) for row in g]
    return "\n".join(header + body)


# =========================
# 表格打印 & CSV 导出
# =========================
def print_table(rows: List[dict]):
    """
    在控制台对齐打印指标表。
    rows 的每项需包含键：
      Algorithm, Path Length (steps), Nodes Expanded, Max Frontier Size, Found Path
    """
    headers = [
        "Algorithm",
        "Path Length (steps)",
        "Nodes Expanded",
        "Max Frontier Size",
        "Found Path",
    ]
    # 计算列宽
    col_w = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            col_w[h] = max(col_w[h], len(str(r[h])))

    # 打印表头
    line = " | ".join(f"{h:<{col_w[h]}}" for h in headers)
    sep = "-+-".join("-" * col_w[h] for h in headers)
    print(line)
    print(sep)
    # 打印行
    for r in rows:
        print(
            " | ".join(
                f"{str(r[h]):<{col_w[h]}}" for h in headers
            )
        )


def write_csv(rows: List[dict], out_csv: str):
    """将指标写入 CSV 文件。"""
    headers = [
        "Algorithm",
        "Path Length (steps)",
        "Nodes Expanded",
        "Max Frontier Size",
        "Found Path",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# =========================
# 统一运行入口
# =========================
def run_all(map_path: str, w_list: List[float]):
    grid, start, goal = load_grid(map_path)
    out_dir = os.path.dirname(os.path.abspath(map_path)) or "."

    results = []

    # 1) 经典四种
    results.append(bfs(grid, start, goal))
    results.append(dfs(grid, start, goal))
    results.append(ucs(grid, start, goal))
    results.append(
        greedy_best_first(grid, start, goal, heuristic=h_manhattan, label="Greedy (Manhattan)")
    )

    # 2) A*（三启发）
    results.append(a_star(grid, start, goal, heuristic=h_manhattan, label="A* (Manhattan)"))
    results.append(a_star(grid, start, goal, heuristic=h_chebyshev, label="A* (Chebyshev)"))
    results.append(a_star(grid, start, goal, heuristic=h_octile, label="A* (Octile)"))

    # 3) Weighted A*（使用 Manhattan；可按需改为其他启发）
    for w in w_list:
        results.append(
            weighted_a_star(
                grid, start, goal, heuristic=h_manhattan, w=w, label=f"Weighted A* w={w:.1f} (Manhattan)"
            )
        )

    # -- 指标整理
    rows = []
    for res in results:
        rows.append(
            {
                "Algorithm": res["algorithm"],
                "Path Length (steps)": res["path_length"],
                "Nodes Expanded": res["nodes_expanded"],
                "Max Frontier Size": res["max_frontier"],
                "Found Path": bool(res["path"]),
            }
        )

    # 控制台打印
    print("\n[Search Metrics]")
    print_table(rows)

    # CSV 导出
    csv_path = os.path.join(out_dir, "search_metrics_extended.csv")
    write_csv(rows, csv_path)
    print(f"\nSaved CSV: {csv_path}")

    # 叠加文本导出
    # 题目要求：必须导出 A*（Manhattan）的叠加为 Astar_path_overlay.txt
    astar_m = next(r for r in results if r["algorithm"] == "A* (Manhattan)")
    overlay_txt = build_overlay_text(grid, astar_m["expanded_set"], astar_m["path"], "A* (Manhattan) Overlay")
    overlay_main_path = os.path.join(out_dir, "Astar_path_overlay.txt")
    with open(overlay_main_path, "w", encoding="utf-8") as f:
        f.write(overlay_txt)
    print(f"Saved Overlay (A* Manhattan, required): {overlay_main_path}")

    # 同时导出其余叠加（便于横向比较）
    def safe_name(name: str) -> str:
        return (
            name.replace("*", "star")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "-")
        )

    for res in results:
        alg = res["algorithm"]
        if alg.startswith("A* (") or alg.startswith("Weighted A*"):
            title = f"{alg} Overlay"
            text = build_overlay_text(grid, res["expanded_set"], res["path"], title)
            path = os.path.join(out_dir, f"{safe_name(alg)}_overlay.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"Saved Overlay: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Search Strategies Lab (with Extended Heuristics & Weighted A*)")
    parser.add_argument(
        "--map",
        type=str,
        default="/Users/fan/Downloads/ailab1/maze-grid.txt",
        help="地图文件路径（默认：/Users/fan/Downloads/ailab1/maze-grid.txt）",
    )
    parser.add_argument(
        "--w",
        type=float,
        nargs="*",
        default=[1.5, 2.0],
        help="Weighted A* 的权重列表（默认：[1.5, 2.0]；每个 >= 1）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all(args.map, args.w)
