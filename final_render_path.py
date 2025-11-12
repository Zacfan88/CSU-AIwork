# -*- coding: utf-8 -*-
# final_path_render.py - Read local maze-grid.txt and output final path
import os
import heapq
import numpy as np
import matplotlib.pyplot as plt

# Use the file in the same directory as this script
BASE_DIR = os.path.dirname(__file__)
MAZE_PATH = os.path.join(BASE_DIR, "maze-grid.txt")

with open(MAZE_PATH, "r", encoding="utf-8") as f:
    rows = [list(line.rstrip("\n")) for line in f]
H, W = len(rows), len(rows[0])

start = None
goal = None
for r in range(H):
    for c in range(W):
        if rows[r][c] == '$':
            start = (r, c)
        if rows[r][c] == '@':
            goal = (r, c)
assert start and goal, "Map must include '$' and '@'"

def passable(r, c):
    return rows[r][c] != '#'

def nbrs(p):
    r, c = p
    for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
        if 0 <= nr < H and 0 <= nc < W and passable(nr, nc):
            yield (nr, nc)

def h(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan

def astar(s, t):
    pq = [(h(s, t), 0, s)]
    came = {s: None}
    g = {s: 0}
    closed = set()
    while pq:
        f, gc, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        if u == t:
            path = []
            x = u
            while x is not None:
                path.append(x)
                x = came[x]
            return list(reversed(path))
        for v in nbrs(u):
            ng = gc + 1
            if v not in g or ng < g[v]:
                g[v] = ng
                came[v] = u
                heapq.heappush(pq, (ng + h(v, t), ng, v))
    return []

path = astar(start, goal)
assert path, "No path found"
print("Path length (steps) =", len(path) - 1)

# Char overlay
overlay = [row[:] for row in rows]
for (r, c) in path:
    if (r, c) != start and (r, c) != goal and overlay[r][c] != '#':
        overlay[r][c] = 'o'
overlay[start[0]][start[1]] = '$'
overlay[goal[0]][goal[1]] = '@'

out_overlay = os.path.join(BASE_DIR, "final_path_overlay.txt")
with open(out_overlay, "w", encoding="utf-8") as f:
    for r in range(H):
        f.write("".join(overlay[r]) + "\n")

# PNG image
A = np.zeros((H, W), float)
for r in range(H):
    for c in range(W):
        A[r, c] = 0 if rows[r][c] == '#' else 1
for (r, c) in path:
    if (r, c) != start and (r, c) != goal:
        A[r, c] = 2
A[start] = 3
A[goal] = 4

plt.figure(figsize=(8, 8))
plt.imshow(A)
plt.title(f"Final Path (A* Manhattan) | steps={len(path)-1}")
plt.xticks([])
plt.yticks([])
plt.grid(which="minor")
out_png = os.path.join(BASE_DIR, "final_path_map.png")
plt.savefig(out_png, bbox_inches="tight", dpi=180)
plt.close()
print(f"Exported: {out_overlay} and {out_png}")