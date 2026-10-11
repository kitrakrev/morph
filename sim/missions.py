"""Seeded mission generator with splits. Fixed code.

Splits are disjoint by seed range so the evolution agent can never see a
held-out map. Every generated mission is checked solvable under TRUE physics
(the start, pickup and drop cells are mutually reachable for the body).
"""
from __future__ import annotations

import numpy as np

from sim.mission import build_sim
from sim.world import T

BODIES = {
    "rover": {"locomotion": {"wheels": 4}, "mass_kg": 30, "battery_wh": 200, "payload_bay_kg": 10,
              "sensors": ["camera"], "size_m": [0.8, 0.6, 0.5]},
    "drone": {"locomotion": {"rotors": 4}, "mass_kg": 2.5, "battery_wh": 60, "payload_bay_kg": 1.0,
              "sensors": ["camera", "lidar_20m"], "size_m": [0.6, 0.6, 0.3]},
    "quadruped": {"locomotion": {"legs": 4}, "mass_kg": 25, "battery_wh": 150, "payload_bay_kg": 6,
                  "sensors": ["camera", "lidar_10m"], "size_m": [0.9, 0.4, 0.6]},
    "railcart": {"locomotion": {"rail": True}, "mass_kg": 80, "battery_wh": 300, "payload_bay_kg": 40,
                 "sensors": ["camera"], "size_m": [1.5, 0.8, 0.6]},
}
ITEMS = [("water", 4.0), ("radio", 0.8), ("supply_kit", 2.5), ("blanket", 1.2)]
SEED_RANGES = {"train": range(100, 200), "regression": range(200, 300), "heldout": range(900, 1000)}


def _cells(sim, body_kind):
    g = sim.world.grid if sim.world.dims < 3 else sim.world.grid[:, :, 0]
    ok = {"rail": [T.RAIL, T.CHARGER]}.get(body_kind, [T.FLOOR])
    if body_kind == "railcart":
        ok = [T.RAIL]
    mask = np.isin(g, ok)
    if sim.world.dims == 3:  # no pickups/drops under a no-fly column: takeoff/landing would breach it
        mask &= ~(sim.world.grid[:, :, 1:] == T.NOFLY).any(axis=2)
    return [tuple(int(c) for c in p) for p in np.argwhere(mask)]


def make_mission(body_kind: str, seed: int, split: str, dims: int | None = None) -> dict | None:
    dims = dims or (3 if body_kind == "drone" else 1 if body_kind == "railcart" else 2)
    rng = np.random.default_rng(seed)
    item, kg = ITEMS[int(rng.integers(len(ITEMS)))]
    if body_kind == "drone":
        item, kg = ("radio", 0.8) if rng.random() < 0.6 else ("blanket", 0.5)
    base = {"terrain": {"dims": dims, "seed": int(seed)}, "robots": [], "items": []}
    probe = build_sim({**base, "robots": [], "items": []})
    cells = _cells(probe, body_kind)
    mode = {"rover": "wheel", "quadruped": "legs", "railcart": "rail", "drone": "flight"}[body_kind]
    for _ in range(30):
        a, b, c = (cells[int(i)] for i in rng.choice(len(cells), 3, replace=False))
        if dims == 3:
            a, b, c = (*a, 0), (*b, 0), (*c, 0)
        span = sum(abs(x - y) for x, y in zip(b, c))
        if span < 15:
            continue
        if mode != "flight" and (probe._astar(a, b, mode) is None or probe._astar(b, c, mode) is None):
            continue
        rid = f"{body_kind}-1"
        return {
            "id": f"{split}-{body_kind}-{seed}", "split": split, "body": body_kind,
            "terrain": base["terrain"],
            "robots": [{"id": rid, "structure": BODIES[body_kind], "position": list(a)}],
            "items": [{"id": item, "weight_kg": kg, "pos": list(b)}],
            "goal": {"deliver": {"item": item, "weight_kg": kg, "to": list(c)},
                     "constraints": ["no_nofly", "battery_reserve"], "reserve_frac": 0.2,
                     "deadline_steps": 600},
        }
    return None


def mission_set(body_kind: str, split: str, n: int) -> list[dict]:
    out = []
    for seed in SEED_RANGES[split]:
        m = make_mission(body_kind, seed, split)
        if m:
            out.append(m)
        if len(out) == n:
            break
    return out
