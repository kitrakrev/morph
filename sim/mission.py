"""Build a Sim from a user-style mission spec (section 3 of the plan). Fixed code."""
from __future__ import annotations

from sim.physics import Body
from sim.skills import Item, Robot, Sim
from sim.world import make_world


def build_sim(mission: dict) -> Sim:
    t = mission["terrain"]
    world = make_world(t["dims"], t.get("size"), t.get("seed", 0), t.get("difficulty", "medium"))
    robots = {}
    for r in mission["robots"]:
        body = Body.from_structure(r["structure"])
        robots[r["id"]] = Robot(r["id"], body, tuple(r["position"]), body.battery_wh,
                                capabilities=r.get("capabilities"))
    items = {i["id"]: Item(i["id"], float(i["weight_kg"]), tuple(i["pos"])) for i in mission.get("items", [])}
    sim = Sim(world, robots, items)
    sim.snapshot()
    return sim
