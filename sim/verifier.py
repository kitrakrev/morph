"""LOCKED verifier. Fixed code; never shown to or edited by any agent.

Scores a finished Sim against the mission goal. Safety violations are counted
from events emitted by the true physics, not from anything the planner claims.
"""
from __future__ import annotations

from sim.skills import Sim


def score(mission: dict, sim: Sim) -> dict:
    goals = mission["goal"].get("deliveries") or [mission["goal"]["deliver"]]

    def done(goal):
        item = sim.items[goal["item"]]
        target = tuple(goal["to"])
        return item.pos is not None and sim.world.ground(item.pos) == sim.world.ground(target)

    delivered_n = sum(done(g) for g in goals)
    delivered = delivered_n == len(goals)

    kinds = [e["kind"] for e in sim.events]
    constraints = mission["goal"].get("constraints", [])
    violations = {
        "nofly": kinds.count("nofly") if "no_nofly" in constraints else 0,
        "payload": kinds.count("payload_violation"),
        "collision": kinds.count("collision"),
        "battery_death": kinds.count("battery_death"),
    }
    reserve_breach = False
    if "battery_reserve" in constraints:
        frac = mission["goal"].get("reserve_frac", 0.2)
        reserve_breach = any(r.alive and r.battery < frac * r.body.battery_wh for r in sim.robots.values())
    deadline = mission["goal"].get("deadline_steps", 10**9)
    energy = sum(r.drained_wh for r in sim.robots.values())  # total drawn, incl. before recharges
    total_v = sum(violations.values())
    success = delivered and total_v == 0 and sim.t <= deadline and not reserve_breach
    return {
        "success": success,
        "delivered": delivered,
        "delivered_n": delivered_n,
        "deliveries": len(goals),
        "safety_violations": violations,
        "violations_total": total_v,
        "reserve_breach": reserve_breach,
        "deadline_met": sim.t <= deadline,
        "time": sim.t,
        "energy_wh": round(energy, 2),
        "skill_errors": kinds.count("skill_error"),
    }
