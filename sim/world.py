"""N-D grid world for Morph. Fixed code: no agent may edit this file.

Grids are numpy int8 arrays of shape `size` (1, 2 or 3 dims). In 3D, z=0 is
the ground layer and z>0 is air unless a building (WALL) or no-fly column.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class T(IntEnum):
    FLOOR = 0
    RUBBLE = 1
    STAIRS = 2
    WATER = 3
    WALL = 4
    AIR = 5
    NOFLY = 6
    CHARGER = 7
    RAIL = 8


Pos = tuple[int, ...]


@dataclass
class World:
    grid: np.ndarray
    seed: int
    chargers: list[Pos] = field(default_factory=list)

    @property
    def dims(self) -> int:
        return self.grid.ndim

    @property
    def size(self) -> tuple[int, ...]:
        return self.grid.shape

    def inside(self, p: Pos) -> bool:
        return len(p) == self.dims and all(0 <= c < s for c, s in zip(p, self.size))

    def at(self, p: Pos) -> T:
        return T(int(self.grid[p]))

    def ground(self, p: Pos) -> Pos:
        """Project a position onto the ground layer (3D only)."""
        return (*p[:2], 0) if self.dims == 3 else p

    def summary(self) -> dict:
        """Public terrain summary the harness may read (counts per type)."""
        vals, counts = np.unique(self.grid, return_counts=True)
        return {"dims": self.dims, "size": list(self.size),
                "terrain": {T(int(v)).name.lower(): int(c) for v, c in zip(vals, counts)},
                "chargers": [list(c) for c in self.chargers]}


def corridor_1d(length: int = 60, seed: int = 0) -> World:
    rng = np.random.default_rng(seed)
    g = np.full((length,), T.RAIL, dtype=np.int8)
    for i in rng.choice(np.arange(5, length - 5), size=length // 10, replace=False):
        g[i] = T.RUBBLE
    chargers = [(length // 2,)]
    g[chargers[0]] = T.CHARGER
    return World(g, seed, chargers)


DIFFICULTY = {  # rubble share of streets, chargers, river fords, no-fly footprint (3D)
    "easy": {"rubble": 0.0, "chargers": 6, "fords": 6, "nofly": (0, 0)},
    "medium": {"rubble": 1 / 12, "chargers": 4, "fords": 3, "nofly": (8, 6)},
    "hard": {"rubble": 1 / 6, "chargers": 2, "fords": 1, "nofly": (12, 10)},
}


def city_2d(w: int = 40, h: int = 40, seed: int = 0, difficulty: str = "medium") -> World:
    """Streets on a block grid, rubble piles, a river, stairs, a rail line, chargers."""
    cfg = DIFFICULTY[difficulty]
    rng = np.random.default_rng(seed)
    g = np.full((w, h), T.FLOOR, dtype=np.int8)
    # buildings: blocks between streets every 8 cells
    for bx in range(2, w - 2, 8):
        for by in range(2, h - 2, 8):
            g[bx:bx + 5, by:by + 5] = T.WALL
    # river across the map with a few fords
    ry = int(rng.integers(h // 3, 2 * h // 3))
    g[:, ry] = T.WATER
    for fx in rng.choice(np.arange(w), size=cfg["fords"], replace=False):
        g[fx, ry] = T.FLOOR
    # rail line along one street
    rx = 1
    g[rx, :] = T.RAIL
    # rubble and stairs scattered on streets
    street = np.argwhere(g == T.FLOOR)
    for idx in rng.choice(len(street), size=int(len(street) * cfg["rubble"]), replace=False):
        g[tuple(street[idx])] = T.RUBBLE
    for idx in rng.choice(len(street), size=len(street) // 40, replace=False):
        g[tuple(street[idx])] = T.STAIRS
    chargers = []
    for idx in rng.choice(len(street), size=cfg["chargers"], replace=False):
        p = tuple(int(c) for c in street[idx])
        g[p] = T.CHARGER
        chargers.append(p)
    return World(g, seed, chargers)


NOFLY_CEIL = 6  # no-fly applies to altitudes 1..5


def city_3d(w: int = 40, h: int = 40, zmax: int = 10, seed: int = 0, difficulty: str = "medium") -> World:
    """2D city as ground layer, buildings extruded to random heights, a no-fly column."""
    base = city_2d(w, h, seed, difficulty)
    fw, fh = DIFFICULTY[difficulty]["nofly"]
    rng = np.random.default_rng(seed + 1)
    g = np.full((w, h, zmax), T.AIR, dtype=np.int8)
    g[:, :, 0] = base.grid
    walls = base.grid == T.WALL
    # one height per building block (blocks are 5x5 on an 8-cell grid)
    for bx in range(2, w - 2, 8):
        for by in range(2, h - 2, 8):
            top = int(rng.integers(3, zmax - 2))
            blk = walls[bx:bx + 5, by:by + 5]
            g[bx:bx + 5, by:by + 5, :top][blk] = T.WALL
    # no-fly zone: a rectangular column (e.g. active fire / helicopter corridor)
    nx, ny = int(rng.integers(w // 4, w // 2)), int(rng.integers(h // 4, h // 2))
    # no-fly zone has a ceiling: restricted below NOFLY_CEIL, open airspace above
    box = g[nx:nx + fw, ny:ny + fh, 1:NOFLY_CEIL]
    g[nx:nx + fw, ny:ny + fh, 1:NOFLY_CEIL] = np.where(box == T.AIR, T.NOFLY, box)
    chargers = [(*c, 0) for c in base.chargers]
    return World(g, seed, chargers)


def make_world(dims: int, size: list[int] | None = None, seed: int = 0, difficulty: str = "medium") -> World:
    if dims == 1:
        return corridor_1d(*(size or [60]), seed=seed)
    if dims == 2:
        return city_2d(*(size or [40, 40]), seed=seed, difficulty=difficulty)
    if dims == 3:
        return city_3d(*(size or [40, 40, 10]), seed=seed, difficulty=difficulty)
    raise ValueError(f"dims must be 1, 2 or 3, got {dims}")


# ---- user-supplied maps -----------------------------------------------------
LEGEND = {".": T.FLOOR, " ": T.FLOOR, "#": T.WALL, "r": T.RUBBLE, "s": T.STAIRS, "~": T.WATER,
          "=": T.RAIL, "C": T.CHARGER}


def _read_map(t: dict) -> list[str]:
    if "map" in t:
        return [str(r) for r in t["map"]]
    path = t["map_file"]
    if path.endswith(".json"):
        import json
        with open(path) as f:
            spec = json.load(f)
        t.update({k: v for k, v in spec.items() if k != "map"})
        return spec["map"]
    if path.endswith((".png", ".jpg", ".jpeg")):  # height map: 0 = street, brighter = taller building
        from PIL import Image
        img = np.asarray(Image.open(path).convert("L"), dtype=float)
        zmax = int(t.get("zmax", 10))
        levels = np.round(img / 255 * (zmax - 2)).astype(int)
        return ["".join(str(min(9, v)) if v > 0 else "." for v in row) for row in levels]
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def world_from_map(t: dict) -> World:
    """World from a user map. `map` is a list of rows (row = y, column = x) using LEGEND;
    digits 1-9 are buildings of that height (3D) or walls (2D). 3D when `zmax` is given or the
    map has digits; `nofly` lists boxes [x0, y0, x1, y1] (optionally + [z0, z1]). One row and
    `dims: 1` gives a corridor."""
    rows = _read_map(t)
    w, h = max(len(r) for r in rows), len(rows)
    has_digits = any(c.isdigit() for r in rows for c in r)
    dims = int(t.get("dims") or (3 if t.get("zmax") or has_digits else (1 if h == 1 else 2)))
    ground = np.full((w, h), T.FLOOR, dtype=np.int8)
    heights = np.zeros((w, h), dtype=int)
    chargers = []
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            if c.isdigit() and c != "0":
                ground[x, y], heights[x, y] = T.WALL, int(c)
            else:
                ground[x, y] = LEGEND.get(c, T.FLOOR)
                if c == "#":
                    heights[x, y] = int(t.get("wall_height", 4))
                if c == "C":
                    chargers.append((x, y))
    seed = int(t.get("seed", 0))
    if dims == 1:
        return World(ground[:, 0].copy(), seed, [(c[0],) for c in chargers])
    if dims == 2:
        return World(ground, seed, chargers)
    zmax = int(t.get("zmax", max(10, heights.max() + 3)))
    g = np.full((w, h, zmax), T.AIR, dtype=np.int8)
    g[:, :, 0] = ground
    for x, y in np.argwhere(heights > 0):
        g[x, y, :min(heights[x, y], zmax - 1)] = T.WALL
    for box in t.get("nofly", []):
        x0, y0, x1, y1, *z = box
        z0, z1 = z if z else (1, NOFLY_CEIL)
        blk = g[x0:x1, y0:y1, z0:z1]
        g[x0:x1, y0:y1, z0:z1] = np.where(blk == T.AIR, T.NOFLY, blk)
    return World(g, seed, [(*c, 0) for c in chargers])


def world_from_spec(t: dict) -> World:
    if "map" in t or "map_file" in t:
        return world_from_map(dict(t))
    return make_world(t["dims"], t.get("size"), t.get("seed", 0), t.get("difficulty", "medium"))
