"""Node-link graph layout (networkx) for the knowledge-graph figure."""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np


MAX_GRAPH_NODES = 28


def _to_networkx(count: int, edges: Sequence[Tuple[int, int]]) -> "nx.Graph":
    """Builds the networkx graph backing every node-link diagram in the lab."""
    G = nx.Graph()
    G.add_nodes_from(range(count))
    G.add_edges_from((s, e) for s, e in edges if 0 <= s < count and 0 <= e < count)
    return G


def _components(G: "nx.Graph") -> List[List[int]]:
    """Connected components (via networkx), sorted largest first."""
    return sorted((sorted(c) for c in nx.connected_components(G)), key=len, reverse=True)


def _force_layout(G: "nx.Graph", nodes: Sequence[int], seed: int = 11) -> np.ndarray:
    """Positions one connected component with networkx's spring_layout
    (Fruchterman-Reingold), normalised into the -1..1 square."""
    count = len(nodes)
    if count == 0:
        return np.zeros((0, 2))
    if count == 1:
        return np.zeros((1, 2))
    if count == 2:
        return np.array([[-0.8, 0.0], [0.8, 0.0]])

    subG = G.subgraph(nodes)
    # A larger-than-default k (ideal spring length) spreads nodes out, since every node
    # carries a text label that must not collide with its neighbours.
    raw = nx.spring_layout(subG, seed=seed, k=2.6 / math.sqrt(count), iterations=400)
    if 3 <= count <= 40:
        # Kamada-Kawai spaces small/medium graphs more evenly than spring layout; the spring
        # result seeds it so the "re-layout" button still produces a different arrangement.
        try:
            raw = nx.kamada_kawai_layout(subG, pos=raw)
        except Exception:
            pass  # scipy missing or non-convergence: keep the spring layout
    pos = np.array([raw[n] for n in nodes])
    return _normalise_layout(pos)


def _normalise_layout(pos: np.ndarray) -> np.ndarray:
    if len(pos) <= 1:
        return np.zeros_like(pos)
    low, high = pos.min(axis=0), pos.max(axis=0)
    span = high - low
    scale = max(span[0], span[1]) or 1.0
    centred = pos - (low + high) / 2.0
    return centred / (scale / 2.0)


def _label_boxes(names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    half_width = np.array([0.022 + 0.006 * len(n[:18]) for n in names], dtype=float)
    half_height = np.full(len(names), 0.060, dtype=float)
    return half_width, half_height


def _separate_labels(pos: np.ndarray, half_width: np.ndarray, half_height: np.ndarray, rounds: int = 140) -> np.ndarray:
    if len(pos) < 2:
        return pos
    pos = pos.copy()
    for _ in range(rounds):
        dx = pos[:, 0][:, None] - pos[:, 0][None, :]
        dy = pos[:, 1][:, None] - pos[:, 1][None, :]
        need_x = half_width[:, None] + half_width[None, :]
        need_y = half_height[:, None] + half_height[None, :]
        ox = need_x - np.abs(dx)
        oy = need_y - np.abs(dy)
        colliding = (ox > 0) & (oy > 0)
        np.fill_diagonal(colliding, False)
        rows, cols = np.where(np.triu(colliding))
        if len(rows) == 0:
            break
        for f, s in zip(rows, cols):
            if ox[f, s] <= oy[f, s]:
                dir_v = np.array([1.0 if dx[f, s] >= 0 else -1.0, 0.0])
                amt = ox[f, s] / 2.0 + 1e-3
            else:
                dir_v = np.array([0.0, 1.0 if dy[f, s] >= 0 else -1.0])
                amt = oy[f, s] / 2.0 + 1e-3
            pos[f] += dir_v * amt
            pos[s] -= dir_v * amt
    return pos


def _graph_layout(names: Sequence[str], edges: Sequence[Tuple[int, int]], seed: int = 11) -> np.ndarray:
    """Node positions computed via networkx: connected components (networkx) are laid
    out independently (spring + Kamada-Kawai), with the largest cluster on top and the
    small ones packed into compact rows beneath it, so unrelated clusters never get
    dragged into one tangle."""
    count = len(names)
    if count <= 1:
        return np.zeros((count, 2))

    G = _to_networkx(count, edges)
    groups = _components(G)
    positions = np.zeros((count, 2))

    local_layouts = [_force_layout(G, grp, seed=seed) for grp in groups]

    if len(groups) == 1:
        for node, pt in zip(groups[0], _normalise_layout(local_layouts[0]) * 0.92):
            positions[node] = pt
    else:
        # The largest cluster gets a big area on top; the many small clusters are packed
        # into compact cells in rows beneath it, instead of being scattered over the canvas.
        for node, pt in zip(groups[0], local_layouts[0]):
            positions[node] = [pt[0] * 0.95, 0.75 + pt[1] * 0.60]
        others = list(zip(groups[1:], local_layouts[1:]))
        # Two columns of generously sized cells (the canvas grows taller instead of
        # squeezing clusters together), so a pair's labels and edge chip never collide.
        ncols = min(len(others), 2 if len(others) <= 6 else 3)
        cell_w, cell_h = 2.0 / ncols, 0.45
        for k, (grp, lay) in enumerate(others):
            r, c = divmod(k, ncols)
            cx = -1.0 + (c + 0.5) * cell_w
            cy = -0.05 - (r + 0.5) * cell_h
            sx, sy = cell_w * 0.40, cell_h * 0.28
            for node, pt in zip(grp, lay):
                positions[node] = [cx + pt[0] * sx, cy + pt[1] * sy]

    hw, hh = _label_boxes(names)
    positions = _separate_labels(positions, hw, hh)
    return positions


def _label_positions(pos: np.ndarray, edges: Sequence[Tuple[int, int]]) -> List[str]:
    count = len(pos)
    neighbors: Dict[int, List[int]] = {i: [] for i in range(count)}
    for s, e in edges:
        neighbors[s].append(e)
        neighbors[e].append(s)

    options = [
        ("bottom center", np.array([0.0, -1.0])),
        ("top center", np.array([0.0, 1.0])),
        ("middle right", np.array([1.0, 0.0])),
        ("middle left", np.array([-1.0, 0.0])),
    ]
    chosen = []
    for i in range(count):
        if not neighbors[i]:
            chosen.append("bottom center")
            continue
        direction = np.zeros(2)
        for nbr in neighbors[i]:
            offset = pos[nbr] - pos[i]
            l = float(np.linalg.norm(offset)) or 1.0
            direction += offset / l
        best = min(options, key=lambda opt: float(np.dot(direction, opt[1])) - (0.25 if "center" in opt[0] else 0.0))
        chosen.append(best[0])
    return chosen
