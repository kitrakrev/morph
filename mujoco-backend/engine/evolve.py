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
    """Hybrid retrieval fused with $rankFusion: autoEmbed text vectors, keyword search, and (when available)
    multimodal vectors of failure map pictures queried cross-modally with the text."""
    pipelines = {
        "vec": [{"$vectorSearch": {"index": "failures_vec", "path": "summary", "query": {"text": query},
                                   "filter": {"body": body}, "limit": 20, "numCandidates": 100}}],
        "kw": [{"$search": {"index": "failures_text", "compound": {
            "must": [{"text": {"query": query, "path": "summary"}}],
            "filter": [{"equals": {"path": "body", "value": body}}]}}}, {"$limit": 20}],
    }
    from engine import multimodal
    if multimodal.enabled():
        try:
            qv = multimodal.embed(query, input_type="query")
            pipelines["mm"] = [{"$vectorSearch": {"index": "failures_mm", "path": "mm_embedding", "queryVector": qv,
                                                  "filter": {"body": body}, "limit": 20, "numCandidates": 100}}]
        except Exception as e:
            print(f"    multimodal query skipped ({str(e)[:60]})")
    pipe = [{"$rankFusion": {"input": {"pipelines": pipelines}}}, {"$limit": k}, {"$project": {"_id": 0, "summary": 1}}]
    try:
        return [d["summary"] for d in db().failures.aggregate(pipe)]
    except Exception as e:  # index still building or no failures yet: say so, never silently
        print(f"    $rankFusion unavailable ({str(e)[:60]}); using recency fallback")
        return [d["summary"] for d in db().failures.find({"body": body}, {"summary": 1}).sort("ts", -1).limit(k)]


RAINDROP_TOOLS = ["mcp__raindrop__raindrop_skills", "mcp__raindrop__list_signals", "mcp__raindrop__list_events",
                  "mcp__raindrop__search_events", "mcp__raindrop__get_event_count"]


def raindrop_evidence_rest(body: str, version: int, key: str) -> dict:
    """Raindrop evidence straight from the Query API (no MCP): signal counts and failure outputs for one
    harness version's train / regression trials. Retries while the trial events are still being indexed."""
    import collections
    import datetime as dt
    import time

    import requests
    since = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {"user_id": body, "convo_id": f"{body}-v{version}", "event_name": "mission_trial", "limit": 100,
              "timestamp[gte]": since, "properties[split][neq]": "heldout"}
    events = []
    for _ in range(4):
        r = requests.get("https://query.raindrop.ai/v1/events", params=params, timeout=30,
                         headers={"Authorization": f"Bearer {key}"})
        r.raise_for_status()
        events = r.json().get("data", [])
        if events:
            break
        time.sleep(20)  # ingestion -> query index lag
    counts = collections.Counter(sig["name"] for e in events for sig in (e.get("signals") or []))
    fails = [(e.get("assistant_output") or "")[:160] for e in events
             if not any("deliver" in sig["name"].lower() for sig in (e.get("signals") or []))]
    return {"source": "raindrop-query-api", "signals": dict(counts), "failures": list(dict.fromkeys(fails))[:3],
            "events_seen": len(events)}


def raindrop_evidence(body: str, version: int) -> dict:
    """Ask Raindrop what it saw for this harness version: Query API when RAINDROP_QUERY_API_KEY is set,
    else its MCP from a headless Claude step."""
    import os
    key = os.getenv("RAINDROP_QUERY_API_KEY")
    if key:
        try:
            return raindrop_evidence_rest(body, version, key)
        except Exception as e:
            print(f"    raindrop query API failed ({str(e)[:80]}); trying MCP")
    prompt = (f"Use the Raindrop MCP tools (call raindrop_skills('explore') first). For user_id '{body}', "
              f"event 'mission_trial', convo_id '{body}-v{version}', last 24h: count matched signals per signal name "
              f"(Battery death, Reserve breach, Nofly violation, Delivered) and quote up to 3 distinct failure outputs "
              f"(first 160 chars). Ignore events whose split property is 'heldout'. "
              f'Reply ONLY JSON: {{"signals": {{name: count}}, "failures": [str], "events_seen": int}}')
    import time
    from engine import telemetry
    telemetry.flush()  # make sure this version's trial events have left the process
    tries, wait = int(os.getenv("MORPH_RAINDROP_TRIES", "3")), int(os.getenv("MORPH_RAINDROP_WAIT", "90"))
    last = {"source": "unavailable"}
    for i in range(tries):
        try:
            out = llm.claude_json("You are a read-only Raindrop analyst.", prompt, model="sonnet", tools=RAINDROP_TOOLS,
                                  cache=False, timeout=240, retries=0)
            last = {"source": "raindrop-mcp", **out["json"]}
        except Exception as e:
            last = {"source": "unavailable", "error": str(e)[:120]}
        if last.get("source") == "raindrop-mcp" and not last.get("error") and (last.get("events_seen") or 0) > 0:
            return last  # Raindrop has indexed this version's events
        if i < tries - 1:
            print(f"    raindrop: no indexed events yet for {body} v{version}; retry in {wait}s ({i + 1}/{tries})")
            time.sleep(wait)
    if last.get("source") == "raindrop-mcp" and not last.get("error"):
        last = {**last, "status": "not_indexed", "signals": None}  # empty because of ingestion delay, not zero
    return last


def propose(parent: dict, parent_eval: dict, rejected: list[dict]) -> dict:
    from engine.planner import SYSTEM as PLANNER_SYSTEM
    fails = [t["diagnosis"] for t in parent_eval["trials"] if not t["score"]["success"]]
    sequences = [{"mission": t["mission_id"], "template": t.get("which"),
                  "actions": " > ".join(c["skill"] for c in t["program"]),
                  "result": "OK" if t["score"]["success"] else t["diagnosis"].split(",")[0]}
                 for t in parent_eval["trials"]]
    past = similar_failures(parent["body_type"], "; ".join(fails[:3]) or "mission failure")
    rd = raindrop_evidence(parent["body_type"], parent["version"])
    print(f"    raindrop: {json.dumps(rd)[:200]}")
    user = json.dumps({
        "harness": {k: parent.get(k) for k in EDITABLE},
        "version": parent["version"],
        "train_result": {"success": parent_eval["success"], "n": parent_eval["n"],
                         "violations": parent_eval["violations"]},
        "current_failures": fails,
        "similar_past_failures": past,
        "raindrop_signals": rd,
        "planner_instructions": PLANNER_SYSTEM,
        "trial_action_sequences": sequences,
        "planner_template": parent_eval["template"]["primary"],
        "planner_fallback": parent_eval["template"]["fallback"],
        "rejected_edits": [{"changes": r["changes"], "reason": r["reason"]} for r in rejected],
    }, default=str)
    from engine.harness import edit_budget
    system = SYSTEM.replace("EDIT_BUDGET", str(edit_budget(parent["version"])))
    try:
        out = llm.claude_json(system, user)
    except Exception as e:  # fall back to the free OpenRouter model
        print(f"    claude -p failed ({str(e)[:80]}); falling back to OpenRouter")
        out = llm.chat_json(system, user, tier="strong", max_tokens=2500)
    edit = out["json"]
    if not out.get("_cached"):
        from engine import telemetry
        telemetry.llm_event("evolution", out["model"], SYSTEM, user, edit, parent["body_type"], parent["version"])
    edit["model"], edit["retrieved"], edit["raindrop"] = out["model"], past, rd
    return edit
