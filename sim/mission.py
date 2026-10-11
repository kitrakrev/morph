"""Build a Sim from a user-style mission spec (section 3 of the plan). Fixed code."""
from __future__ import annotations

import os

from sim.physics import Body
from sim.skills import Item, Robot, Sim
from sim.world import world_from_spec


def build_sim(mission: dict) -> Sim:
    t = mission["terrain"]
    world = world_from_spec(t)
    robots = {}
    for r in mission["robots"]:
        body = Body.from_structure(r["structure"])
        robots[r["id"]] = Robot(r["id"], body, tuple(r["position"]), body.battery_wh,
                                capabilities=r.get("capabilities"))
    items = {i["id"]: Item(i["id"], float(i["weight_kg"]), tuple(i["pos"])) for i in mission.get("items", [])}
    backend = mission.get("physics") or os.environ.get("MORPH_PHYSICS", "mujoco")
    if backend == "mujoco":
        from sim.mujoco_backend import MujocoSim
        sim = MujocoSim(world, robots, items)
        sim.__dict__["items0"] = {i: it.pos for i, it in items.items()}  # start poses for render3d
    else:
        sim = Sim(world, robots, items)
    sim.snapshot()
    return sim
