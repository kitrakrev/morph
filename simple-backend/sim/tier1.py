"""Tier 1: fast feasibility check (<1 s) using the HARNESS's beliefs, not true physics.

Path existence uses the public map. Energy uses the harness's believed
Wh-per-cell, so a wrong belief can pass tier 1 and still fail in tier 2.
That gap is evidence for the evolution agent.
"""
from __future__ import annotations

from sim.mission import build_sim
from sim.skills import PRIMITIVES
from sim.world import T

DEFAULT_BELIEFS = {"wh_per_cell": {"wheel": 0.15, "legs": 0.35, "rail": 0.05, "flight": 0.4, "fixed_wing": 0.25},
                   "payload_kg_max": None, "reserve_frac": 0.0}


def check(mission: dict, program: list[dict], beliefs: dict | None = None) -> dict:
    b = {**DEFAULT_BELIEFS, **(beliefs or {})}
    sim = build_sim(mission)
    reasons: list[str] = []
    est = {rid: 0.0 for rid in sim.robots}
    pos = {rid: r.pos for rid, r in sim.robots.items()}
    load = {rid: 0.0 for rid in sim.robots}
    for i, call in enumerate(program):
        rid, name, args = call.get("robot"), call.get("skill"), call.get("args", {})
        if name not in PRIMITIVES:
            reasons.append(f"step {i}: unknown skill {name}")
            continue
        if rid not in sim.robots:
            reasons.append(f"step {i}: unknown robot {rid}")
            continue
        r = sim.robots[rid]
        if name in ("move_to", "fly_to"):
            air = "fixed_wing" if "fixed_wing" in r.body.modes else "flight"
            mode = air if name == "fly_to" else next((m for m in r.body.modes if m not in ("flight", "fixed_wing")), None)
            if mode is None or mode not in r.body.modes:
                reasons.append(f"step {i}: {rid} cannot {name}")
                continue
            goal = tuple(args.get("pos", ()))
            if sim.world.dims == 3 and len(goal) == 2:
                goal = (*goal, 0)
            if mode in ("flight", "fixed_wing") and sim.world.dims == 3:
                alt = max(1, int(args.get("alt", 1)))
                avoid = (T.NOFLY,) if args.get("avoid_nofly") else ()
                path = sim._astar((*pos[rid][:2], alt), (*goal[:2], alt), "flight", layer=alt, avoid=avoid)
                if path and any(sim.world.at(p) == T.NOFLY for p in path):
                    reasons.append(f"step {i}: route crosses no-fly at alt {alt}")
                n = (len(path) + 2 * alt) if path is not None else None
            else:
                path = sim._astar(pos[rid], goal, mode)
                n = len(path) if path is not None else None
            if n is None:
                reasons.append(f"step {i}: no {mode} path to {list(goal)}")
                continue
            est[rid] += n * b["wh_per_cell"].get(mode, 0.5)
            pos[rid] = goal
        elif name == "pick":
            it = sim.items.get(args.get("item"))
            if it is None:
                reasons.append(f"step {i}: unknown item")
                continue
            load[rid] += it.weight_kg
            cap = b["payload_kg_max"] if b["payload_kg_max"] is not None else r.body.payload_max_kg
            if load[rid] > cap:
                reasons.append(f"step {i}: payload {load[rid]:.1f}kg > {cap}kg")
        elif name == "charge":
            if sim.world.at(sim.world.ground(pos[rid])) != T.CHARGER:
                reasons.append(f"step {i}: charge called off-charger")
            est[rid] = 0.0
    for rid, wh in est.items():
        cap = sim.robots[rid].body.battery_wh * (1 - b["reserve_frac"])
        if wh > cap:
            reasons.append(f"{rid}: est {wh:.1f}Wh exceeds usable {cap:.1f}Wh")
    return {"pass": not reasons, "reasons": reasons, "est_wh": {k: round(v, 2) for k, v in est.items()}}
