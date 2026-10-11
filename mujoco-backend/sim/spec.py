"""User mission specs: any robot + any map + any goal, as one JSON file. Fixed code.

    {
      "name": "go2-plant-inspection",
      "robot":   {"model_file": "sim/assets/menagerie/unitree_go2/go2.xml", "battery_wh": 230},
      "terrain": {"map_file": "specs/maps/plant.txt"}            # or {"map": [...]} or {"dims": 3, "difficulty": "hard"}
      "start":   "random" | [x, y(, z)],
      "items":   [{"id": "kit", "weight_kg": 2, "pos": "random"}],
      "goal": {"tasks": [{"type": "inspect", "pos": "random"}, {"type": "return_home"}],
               "constraints": ["battery_reserve"], "reserve_frac": 0.2, "deadline_steps": 600}
    }

`register(spec)` makes the spec a body the harness loop can evolve on: `mission_set(name, split, n)`
returns seeded variants (every "random" position re-drawn per seed, generated terrain re-seeded), and
train / regression / held-out splits stay disjoint by seed range exactly like the built-in matrix.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from sim.mission import build_sim
from sim.physics import Body
from sim.robot_import import resolve
from sim.world import T

SPECS: dict[str, dict] = {}


def load(path: str) -> dict:
    with open(path) as f:
        spec = json.load(f)
    return register(spec)


def register(spec: dict) -> dict:
    from sim import missions
    spec = copy.deepcopy(spec)
    spec["robot"] = resolve(spec["robot"])
    body = Body.from_structure(spec["robot"])
    name = spec["name"]
    SPECS[name] = spec
    missions.BODIES[name] = spec["robot"]
    missions.MODE[name] = next((m for m in body.modes if m not in ("flight", "fixed_wing")), "flight")
    if "map" not in spec["terrain"] and "map_file" not in spec["terrain"]:
        missions.DEFAULT_DIMS[name] = spec["terrain"].get("dims", 2)
    return spec


def _cells(sim, body: Body) -> list[tuple]:
    g = sim.world.grid if sim.world.dims < 3 else sim.world.grid[:, :, 0]
    ok = (T.RAIL, T.CHARGER) if body.modes == ("rail",) else (T.FLOOR,)
    mask = np.isin(g, ok)
    if sim.world.dims == 3:
        mask &= ~(sim.world.grid[:, :, 1:] == T.NOFLY).any(axis=2)
    lift = (lambda p: (*p, 0)) if sim.world.dims == 3 else (lambda p: p)
    return [lift(tuple(int(c) for c in p)) for p in np.argwhere(mask)]


def make(name: str, seed: int, split: str) -> dict | None:
    spec = SPECS[name]
    rng = np.random.default_rng(seed)
    terrain = dict(spec["terrain"])
    if "map" not in terrain and "map_file" not in terrain:
        terrain["seed"] = int(seed)
    body = Body.from_structure(spec["robot"])
    probe = build_sim({"terrain": terrain, "robots": [], "items": [], "physics": "grid"})
    cells = _cells(probe, body)
    if len(cells) < 3:
        return None
    mode = next((m for m in body.modes if m not in ("flight", "fixed_wing")), None)
    air = mode is None

    def reach(a, b):
        return air or probe._astar(tuple(a), tuple(b), mode) is not None

    for _ in range(40):
        start = spec.get("start", "random")
        start = list(cells[int(rng.integers(len(cells)))]) if start == "random" else list(start)

        def pick():
            return list(cells[int(rng.integers(len(cells)))])
        fixed = lambda v: pick() if v == "random" else v  # noqa: E731
        items = [{**i, "pos": fixed(i["pos"])} for i in spec.get("items", [])]
        goal = copy.deepcopy(spec["goal"])
        for t in goal.get("tasks", []):
            for k in ("pos", "to"):
                if k in t:
                    t[k] = fixed(t[k])
            if "points" in t:
                t["points"] = [fixed(p) for p in t["points"]]
        pts = [i["pos"] for i in items] + [t[k] for t in goal.get("tasks", []) for k in ("pos", "to") if k in t] + \
              [p for t in goal.get("tasks", []) for p in t.get("points", [])]
        if not all(reach(start, p) for p in pts):
            continue
        rid = f"{name}-1"
        return {"id": f"{split}-{name}-{seed}", "split": split, "body": name, "difficulty": "spec", "mtype": "spec",
                "terrain": terrain, "robots": [{"id": rid, "structure": spec["robot"], "position": start}],
                "items": items, "goal": {"constraints": [], **goal}}
    return None
