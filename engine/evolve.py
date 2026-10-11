"""Evolution agent: reads failures from Atlas, proposes ONE bounded harness edit with a prediction.

Never sees held-out missions: failures/episodes only come from train/regression runs.
"""
from __future__ import annotations

import json

from engine import llm
from engine.harness import EDITABLE
from engine.store import db

SYSTEM = """You improve the HARNESS around a fixed robot planner model. You never change the model,
physics, verifier or engine code. Output ONLY JSON.
Propose ONE edit made of at most EDIT_BUDGET field changes:
{"changes": [{"field": <one of %s>, "value": <complete new value for that field>}, ...],
 "prediction": {"train_success_delta": <int>}, "rationale": "<one sentence citing failures>"}
Facts about fields:
- workflow.nodes may contain: plan, tier1_check, replan, tier2_sim, verify (plan, tier2_sim, verify required).
  tier1_check estimates battery from beliefs; replan switches to the planner's safer fallback when tier1 fails
  (needs budgets.replans >= 1).
- beliefs.wh_per_cell is the harness's energy-per-cell estimate used by tier1; beliefs.reserve_frac reserve it plans for.
- tool_specs.fly_to: {"alt": [min,max], "avoid_nofly": bool}. rules are text the planner must follow.
- retrieval_policy.episodes / procedures: how many memories the planner reads.
- Locked: constraints.nofly and constraints.collision must stay "hard". Tools must be primitives.
Do not repeat a previously rejected edit.""" % (list(EDITABLE),)


def similar_failures(body: str, query: str, k: int = 6) -> list[str]:
    """Hybrid retrieval: autoEmbed vector search + keyword search fused with $rankFusion."""
    pipe = [{"$rankFusion": {"input": {"pipelines": {
        "vec": [{"$vectorSearch": {"index": "failures_vec", "path": "summary", "query": {"text": query},
                                   "filter": {"body": body}, "limit": 20, "numCandidates": 100}}],
        "kw": [{"$search": {"index": "failures_text", "compound": {
            "must": [{"text": {"query": query, "path": "summary"}}],
            "filter": [{"equals": {"path": "body", "value": body}}]}}}, {"$limit": 20}],
    }}}}, {"$limit": k}, {"$project": {"_id": 0, "summary": 1}}]
    try:
        return [d["summary"] for d in db().failures.aggregate(pipe)]
    except Exception:  # index still building or no failures yet
        return [d["summary"] for d in db().failures.find({"body": body}, {"summary": 1}).sort("ts", -1).limit(k)]


def propose(parent: dict, parent_eval: dict, rejected: list[dict]) -> dict:
    fails = [t["diagnosis"] for t in parent_eval["trials"] if not t["score"]["success"]]
    past = similar_failures(parent["body_type"], "; ".join(fails[:3]) or "mission failure")
    user = json.dumps({
        "harness": {k: parent.get(k) for k in EDITABLE},
        "version": parent["version"],
        "train_result": {"success": parent_eval["success"], "n": parent_eval["n"],
                         "violations": parent_eval["violations"]},
        "current_failures": fails,
        "similar_past_failures": past,
        "planner_template": parent_eval["template"]["primary"],
        "planner_fallback": parent_eval["template"]["fallback"],
        "rejected_edits": [{"changes": r["changes"], "reason": r["reason"]} for r in rejected],
    }, default=str)
    from engine.harness import edit_budget
    out = llm.chat_json(SYSTEM.replace("EDIT_BUDGET", str(edit_budget(parent["version"]))), user,
                        tier="strong", max_tokens=2500)
    edit = out["json"]
    if not out.get("_cached"):
        from engine import telemetry
        telemetry.llm_event("evolution", out["model"], SYSTEM, user, edit, parent["body_type"], parent["version"])
    edit["model"], edit["retrieved"] = out["model"], past
    return edit
