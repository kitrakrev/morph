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


def city_2d(w: int = 40, h: int = 40, seed: int = 0) -> World:
    """Streets on a block grid, rubble piles, a river, stairs, a rail line, chargers."""
    rng = np.random.default_rng(seed)
    g = np.full((w, h), T.FLOOR, dtype=np.int8)
    # buildings: blocks between streets every 8 cells
    for bx in range(2, w - 2, 8):
        for by in range(2, h - 2, 8):
            g[bx:bx + 5, by:by + 5] = T.WALL
    # river across the map with a few fords
    ry = int(rng.integers(h // 3, 2 * h // 3))
    g[:, ry] = T.WATER
    for fx in rng.choice(np.arange(w), size=3, replace=False):
        g[fx, ry] = T.FLOOR
    # rail line along one street
    rx = 1
    g[rx, :] = T.RAIL
    # rubble and stairs scattered on streets
    street = np.argwhere(g == T.FLOOR)
    for idx in rng.choice(len(street), size=len(street) // 12, replace=False):
        g[tuple(street[idx])] = T.RUBBLE
    for idx in rng.choice(len(street), size=len(street) // 40, replace=False):
        g[tuple(street[idx])] = T.STAIRS
    chargers = []
    for idx in rng.choice(len(street), size=4, replace=False):
        p = tuple(int(c) for c in street[idx])
        g[p] = T.CHARGER
        chargers.append(p)
    return World(g, seed, chargers)


def city_3d(w: int = 40, h: int = 40, zmax: int = 10, seed: int = 0) -> World:
    """2D city as ground layer, buildings extruded to random heights, a no-fly column."""
    base = city_2d(w, h, seed)
    rng = np.random.default_rng(seed + 1)
    g = np.full((w, h, zmax), T.AIR, dtype=np.int8)
    g[:, :, 0] = base.grid
    walls = base.grid == T.WALL
    heights = rng.integers(3, zmax - 2, size=(w, h))
    for x, y in np.argwhere(walls):
        g[x, y, : heights[x, y]] = T.WALL
    # no-fly zone: a rectangular column (e.g. active fire / helicopter corridor)
    nx, ny = int(rng.integers(w // 4, w // 2)), int(rng.integers(h // 4, h // 2))
    g[nx:nx + 8, ny:ny + 6, 1:] = np.where(g[nx:nx + 8, ny:ny + 6, 1:] == T.AIR, T.NOFLY,
                                           g[nx:nx + 8, ny:ny + 6, 1:])
    chargers = [(*c, 0) for c in base.chargers]
    return World(g, seed, chargers)


def make_world(dims: int, size: list[int] | None = None, seed: int = 0) -> World:
    if dims == 1:
        return corridor_1d(*(size or [60]), seed=seed)
    if dims == 2:
        return city_2d(*(size or [40, 40]), seed=seed)
    if dims == 3:
        return city_3d(*(size or [40, 40, 10]), seed=seed)
    raise ValueError(f"dims must be 1, 2 or 3, got {dims}")
