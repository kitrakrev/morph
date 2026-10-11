"""Tier 2: full simulation of a mission program against TRUE physics. Fixed code."""
from __future__ import annotations

from sim.mission import build_sim
from sim.skills import MAX_STEPS, Sim


def simulate(mission: dict, program: list[dict]) -> Sim:
    sim = build_sim(mission)
    for call in program:
        if sim.t >= MAX_STEPS:
            sim.emit("step_limit", call.get("robot", "?"))
            break
        res = sim.run_call(call)
        sim.emit("call", call.get("robot", "?"), skill=call.get("skill"), ok=res.get("ok"),
                 reason=res.get("reason"))
    return sim
