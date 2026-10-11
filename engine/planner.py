"""Planner: one LLM call per (harness version, body) writes a plan TEMPLATE.

The template uses symbols the compiler fills per mission, so one cheap call
covers every mission and evaluation is deterministic. The planner also returns
a fallback template that the `replan` workflow node switches to.
"""
from __future__ import annotations

import json

from sim.mission import build_sim
from engine import llm
from engine.memory import Memory

SYMBOLS = {"$ROBOT": "robot id", "$ITEM": "item id", "$ITEM_POS": "item position",
           "$DEST": "delivery position", "$CHARGER": "charger nearest the pickup->dest route"}

PLANNER_MODEL = "haiku"

SYSTEM = """You are the mission planner inside a robot-fleet harness. Output ONLY JSON.
Write a reusable plan TEMPLATE for one robot body. Each step: {"skill": <primitive>, "args": {...}}.
Primitives: move_to(pos), fly_to(pos, alt, avoid_nofly), pick(item), drop(item), charge(), scan().
Use symbols instead of coordinates: $ITEM, $ITEM_POS, $DEST, $CHARGER.
Follow every harness rule and stay inside tool_specs ranges.
Return {"primary": [steps], "fallback": [steps], "rationale": "<one sentence>"}.
The fallback is used if the primary fails a feasibility check (e.g. battery); make it safer."""


def context(h: dict, body_kind: str) -> dict:
    mem = Memory(body_kind)
    rp = h.get("retrieval_policy", {})
    ctx = {}
    if rp.get("episodes"):
        ctx["past_missions"] = [e["content"] for e in mem.search_episodes("mission failure battery no-fly",
                                                                          rp["episodes"])]
    if rp.get("procedures"):
        ctx["procedures"] = [p["content"] for p in mem.discover_procedures("deliver item safely", rp["procedures"])]
    if rp.get("same_body_category") and rp.get("episodes"):
        ctx["body_categories"] = [t["definition"] for t in Memory.search_taxonomic(body_kind, 2)]
    return ctx


def plan_template(h: dict, body: dict, terrain_summary: dict | None = None) -> dict:
    harness_view = {k: h.get(k) for k in ("rules", "constraints", "tool_specs", "skills")}
    reports = {}
    if "terrain_summary" in h.get("reports_read", []) and terrain_summary:
        reports["terrain_summary"] = terrain_summary
    user = json.dumps({"body": body, "harness": harness_view, "reports": reports,
                       "memory": context(h, body.get("kind", "?")), "symbols": SYMBOLS})
    # planner model is FIXED for the whole day (Claude Haiku via Claude Code headless); no fallback
    out = llm.claude_json(SYSTEM, user, model=PLANNER_MODEL)
    tpl = out["json"]
    if not out.get("_cached"):
        from engine import telemetry
        telemetry.llm_event("planner", out["model"], SYSTEM, user, tpl, body.get("kind", "?"), h.get("version"))
    return {"primary": tpl.get("primary", []), "fallback": tpl.get("fallback") or tpl.get("primary", []),
            "rationale": tpl.get("rationale", ""), "model": out["model"], "cached": out.get("_cached", False)}


def _sub(v, env):
    if isinstance(v, str) and v in env:
        return env[v]
    if isinstance(v, list):
        return [_sub(x, env) for x in v]
    if isinstance(v, dict):
        return {k: _sub(x, env) for k, x in v.items()}
    return v


def compile_program(template: list[dict], mission: dict, h: dict) -> list[dict]:
    """Fill symbols for one mission and clamp args to the harness tool_specs."""
    sim = build_sim(mission)
    r = mission["robots"][0]
    it = mission["items"][0]
    dest = mission["goal"]["deliver"]["to"]
    route = lambda c: sum(abs(a - b) for a, b in zip(c, it["pos"])) + sum(abs(a - b) for a, b in zip(c, dest))
    charger = list(min(sim.world.chargers, key=route)) if sim.world.chargers else it["pos"]
    env = {"$ROBOT": r["id"], "$ITEM": it["id"], "$ITEM_POS": it["pos"], "$DEST": dest, "$CHARGER": charger}
    prog = []
    for step in template:
        call = {"robot": r["id"], "skill": step.get("skill"), "args": _sub(step.get("args", {}), env)}
        spec = h.get("tool_specs", {}).get(call["skill"], {})
        if call["skill"] == "fly_to":
            lo, hi = spec.get("alt", [1, 8])
            call["args"]["alt"] = max(lo, min(hi, int(call["args"].get("alt", lo))))
            if spec.get("avoid_nofly"):
                call["args"]["avoid_nofly"] = True
        if call["skill"] in ("pick", "drop") and "item" not in call["args"]:
            call["args"]["item"] = it["id"]
        if call["skill"] in ("charge", "scan"):
            call["args"] = {}
        prog.append(call)
    return prog
