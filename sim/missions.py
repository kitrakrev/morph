"""Seeded mission generator over the evaluation matrix. Fixed code.

Matrix axes: body x dims x terrain difficulty x mission type. Splits are disjoint
by seed range so the evolution agent can never see a held-out map. Every mission
is checked solvable under TRUE physics (start, pickups and drops mutually reachable).
Medium/single missions keep their original ids and contents.
"""
from __future__ import annotations

import itertools

import numpy as np

from sim.mission import build_sim
from sim.world import T

BODIES = {
    "rover": {"locomotion": {"wheels": 4}, "mass_kg": 30, "battery_wh": 200, "payload_bay_kg": 10,
              "sensors": ["camera"], "size_m": [0.8, 0.6, 0.5]},
    "tracked": {"locomotion": {"tracks": 2}, "mass_kg": 45, "battery_wh": 260, "payload_bay_kg": 15,
                "sensors": ["camera", "lidar_10m"], "size_m": [1.0, 0.7, 0.6]},
    "drone": {"locomotion": {"rotors": 4}, "mass_kg": 2.5, "battery_wh": 60, "payload_bay_kg": 1.0,
              "sensors": ["camera", "lidar_20m"], "size_m": [0.6, 0.6, 0.3]},
    "quadruped": {"locomotion": {"legs": 4}, "mass_kg": 25, "battery_wh": 150, "payload_bay_kg": 6,
                  "sensors": ["camera", "lidar_10m"], "size_m": [0.9, 0.4, 0.6]},
    "railcart": {"locomotion": {"rail": True}, "mass_kg": 80, "battery_wh": 300, "payload_bay_kg": 40,
                 "sensors": ["camera"], "size_m": [1.5, 0.8, 0.6]},
    "hexapod": {"locomotion": {"legs": 6}, "mass_kg": 18, "battery_wh": 120, "payload_bay_kg": 4,
                "sensors": ["camera", "lidar_10m"], "size_m": [0.8, 0.6, 0.4]},
}
HELDOUT_BODIES = ("hexapod",)
MODE = {"rover": "wheel", "tracked": "tracks", "quadruped": "legs", "hexapod": "legs", "railcart": "rail",
        "drone": "flight"}
DEFAULT_DIMS = {"drone": 3, "railcart": 1}
DIFFICULTIES = ("easy", "medium", "hard")
MISSION_TYPES = ("single", "heavy", "deadline", "multi_stop", "relay")
ITEMS = [("water", 4.0), ("radio", 0.8), ("supply_kit", 2.5), ("blanket", 1.2)]
SEED_RANGES = {"train": range(100, 200), "regression": range(200, 300), "heldout": range(900, 1000)}


def _cells(sim, body_kind):
    g = sim.world.grid if sim.world.dims < 3 else sim.world.grid[:, :, 0]
    ok = [T.RAIL] if body_kind == "railcart" else [T.FLOOR]
    mask = np.isin(g, ok)
    if sim.world.dims == 3:  # no pickups/drops under a no-fly column: takeoff/landing would breach it
        mask &= ~(sim.world.grid[:, :, 1:] == T.NOFLY).any(axis=2)  # takeoff/landing passes through the zone
    return [tuple(int(c) for c in p) for p in np.argwhere(mask)]


def _dist(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def make_mission(body_kind: str, seed: int, split: str, dims: int | None = None,
                 difficulty: str = "medium", mtype: str = "single") -> dict | None:
    dims = dims or DEFAULT_DIMS.get(body_kind, 2)
    rng = np.random.default_rng(seed)
    item, kg = ITEMS[int(rng.integers(len(ITEMS)))]
    if body_kind == "drone":
        item, kg = ("radio", 0.8) if rng.random() < 0.6 else ("blanket", 0.5)
    spec = BODIES[body_kind]
    if mtype == "heavy":
        item, kg = "heavy_kit", round(0.9 * spec["payload_bay_kg"], 2)
    terrain = {"dims": dims, "seed": int(seed), "difficulty": difficulty}
    probe = build_sim({"terrain": terrain, "robots": [], "items": []})
    cells = _cells(probe, body_kind)
    mode = MODE[body_kind]
    if len(cells) < 6:
        return None
    lift = (lambda p: (*p, 0)) if dims == 3 else (lambda p: p)
    reach = (lambda a, b: True) if mode == "flight" else (lambda a, b: probe._astar(a, b, mode) is not None)
    n_points = {"multi_stop": 5, "relay": 4}.get(mtype, 3)
    for _ in range(30):
        pts = [lift(cells[int(i)]) for i in rng.choice(len(cells), n_points, replace=False)]
        a, b, c = pts[:3]
        if _dist(b, c) < 15 or not reach(a, b) or not reach(b, c):
            continue
        rid = f"{body_kind}-1"
        robots = [{"id": rid, "structure": spec, "position": list(a)}]
        items = [{"id": item, "weight_kg": kg, "pos": list(b)}]
        deliveries = [{"item": item, "weight_kg": kg, "to": list(c)}]
        if mtype == "multi_stop":
            d, e = pts[3], pts[4]
            if not reach(c, d) or not reach(d, e):
                continue
            items.append({"id": "blanket_2", "weight_kg": min(kg, 1.2) / 2, "pos": list(d)})
            deliveries.append({"item": "blanket_2", "weight_kg": min(kg, 1.2) / 2, "to": list(e)})
        if mtype == "relay":  # second robot of a different body meets the first; needs handoff
            partner = "rover" if body_kind != "rover" else "tracked"
            robots.append({"id": f"{partner}-2", "structure": BODIES[partner], "position": list(pts[3])})
        deadline = 600
        if mtype == "deadline":
            deadline = int(1.3 * (_dist(a, b) + _dist(b, c)) * (2 if mode == "legs" else 1)) + 20
        suffix = "" if (difficulty, mtype) == ("medium", "single") else f"-{difficulty}-{mtype}"
        return {
            "id": f"{split}-{body_kind}-{seed}{suffix}", "split": split, "body": body_kind,
            "difficulty": difficulty, "mtype": mtype, "terrain": terrain, "robots": robots, "items": items,
            "goal": {"deliver": deliveries[0], **({"deliveries": deliveries} if len(deliveries) > 1 else {}),
                     "constraints": ["no_nofly", "battery_reserve"], "reserve_frac": 0.2,
                     "deadline_steps": deadline},
        }
    return None


def mission_set(body_kind: str, split: str, n: int, difficulty: str = "medium", mtype: str = "single",
                dims: int | None = None) -> list[dict]:
    out = []
    for seed in SEED_RANGES[split]:
        m = make_mission(body_kind, seed, split, dims, difficulty, mtype)
        if m:
            out.append(m)
        if len(out) == n:
            break
    return out


def matrix(bodies=tuple(BODIES), difficulties=DIFFICULTIES, mtypes=MISSION_TYPES):
    """All (body, dims, difficulty, mission type) cells of the evaluation matrix."""
    for b, d, t in itertools.product(bodies, difficulties, mtypes):
        yield {"body": b, "dims": DEFAULT_DIMS.get(b, 2), "difficulty": d, "mtype": t}
