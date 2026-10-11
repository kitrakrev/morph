"""LOCKED verifier. Fixed code; never shown to or edited by any agent.

Scores a finished Sim against the mission goal. Safety violations are counted
from events emitted by the true physics, not from anything the planner claims.
"""
from __future__ import annotations

from sim.goals import check, tasks
from sim.skills import Sim


def score(mission: dict, sim: Sim) -> dict:
    all_tasks = tasks(mission["goal"])
    done = [check(sim, t) for t in all_tasks]
    deliver = [d for d, t in zip(done, all_tasks) if t["type"] == "deliver"]
    delivered_n = sum(deliver)
    goals = deliver
    goal_met = all(done)
    delivered = goal_met  # "delivered" = every task met (kept for older readers)

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
    success = goal_met and total_v == 0 and sim.t <= deadline and not reserve_breach
    return {
        "success": success,
        "delivered": delivered,
        "delivered_n": delivered_n,
        "deliveries": len(goals),
        "tasks_done": sum(done),
        "tasks": len(all_tasks),
        "task_status": [{"type": t["type"], "done": d} for t, d in zip(all_tasks, done)],
        "safety_violations": violations,
        "violations_total": total_v,
        "reserve_breach": reserve_breach,
        "deadline_met": sim.t <= deadline,
        "time": sim.t,
        "energy_wh": round(energy, 2),
        "skill_errors": kinds.count("skill_error"),
    }
