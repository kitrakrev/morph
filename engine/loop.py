"""Evolution loop: generations of propose -> evaluate -> gate -> commit.

Usage: PYTHONPATH=. uv run python -m engine.loop drone --gens 4
"""
from __future__ import annotations

import argparse
import json

from engine import harness
from engine.evolve import propose
from engine.gate import commit, decide
from engine.llm import BudgetExhausted
from engine.runner import evaluate
from engine.store import db
from sim.missions import BODIES, mission_set


def _brief(e: dict) -> str:
    return f"{e['success']}/{e['n']} (n={e['n']}) viol={e['violations']}"


def adapt_curve(body: str, lineage: str, start: str = "cold", gens: int = 3, target: float = 0.8) -> dict:
    """Generations until held-out success >= target, for a cold or warm (transfer) start."""
    from engine import transfer
    heldout = mission_set(body, "heldout", 12)
    if not harness.latest(lineage):
        if start == "warm":
            h, src = transfer.warm_bootstrap(BODIES[body], lineage)
            print(f"[{lineage}] warm start from {src}")
        else:
            h = harness.bootstrap(BODIES[body])
            h["body_type"] = lineage
        harness.save(h)
    curve = []
    for g in range(gens + 1):
        h = harness.latest(lineage)
        ho = evaluate(h, heldout)
        curve.append({"gen": g, "version": h["version"], "heldout": ho["success"], "n": ho["n"],
                      "viol": ho["violations"]})
        print(f"[{lineage}] gen {g}: v{h['version']} held-out {ho['success']}/{ho['n']} (n={ho['n']}) viol={ho['violations']}")
        if ho["success"] >= target * ho["n"] or g == gens:
            break
        before = h["version"]
        run(body, 1, lineage=lineage)
        if harness.latest(lineage)["version"] == before:  # rejected: one more proposal counts as a generation
            continue
    reached = next((c["gen"] for c in curve if c["heldout"] >= target * c["n"]), None)
    return {"lineage": lineage, "start": start, "curve": curve, "gens_to_adapt": reached}


def run(body: str, gens: int, n_train: int = 6, n_reg: int = 4, max_rejects: int = 3,
        mtype: str = "single", difficulty: str = "medium", lineage: str | None = None) -> list[dict]:
    lineage = lineage or body
    train = mission_set(body, "train", n_train, difficulty, mtype)
    # regression always guards the original task (medium / single), so a new skill can't break the old one
    reg = mission_set(body, "regression", n_reg)
    parent = harness.latest(lineage)
    if not parent:
        h = harness.bootstrap(BODIES[body])
        h["body_type"] = lineage
        parent = harness.save(h)
    p_train, p_reg = evaluate(parent, train), evaluate(parent, reg)
    print(f"[{lineage}] v{parent['version']} train {_brief(p_train)} reg {_brief(p_reg)}")
    log, rejected, streak = [], [], 0
    next_version = (db().harnesses.find_one({"body_type": lineage}, sort=[("version", -1)]) or parent)["version"] + 1
    for g in range(gens):
        if p_train["success"] == p_train["n"]:
            print(f"[{body}] train saturated; stopping")
            break
        try:
            edit = propose(parent, p_train, rejected)
        except BudgetExhausted as e:
            print("budget:", e)
            break
        try:
            cand = harness.apply_edit(parent, edit)
            cand["version"] = next_version
        except (ValueError, KeyError) as e:
            rejected.append({"changes": edit.get("changes"), "reason": str(e)})
            print(f"  gen {g + 1}: illegal edit: {e}")
            continue
        next_version += 1
        bad = harness.validate(cand)
        if bad:
            c_train = c_reg = {"success": -1, "n": n_train, "violations": 0, "trials": []}
            decision = {"accepted": False, "checks": {"legal_edit": False}, "reason": "illegal: " + "; ".join(bad),
                        "metrics": {"parent": {"train": p_train["success"]}, "candidate": {"train": -1}}}
        else:
            c_train, c_reg = evaluate(cand, train), evaluate(cand, reg)
            decision = decide(parent, cand, p_train, c_train, p_reg, c_reg)
        ev = commit(cand, decision, edit)
        for c in cand["diff"]:
            print(f"  gen {g + 1}: v{cand['version']} {c['field']} -> {json.dumps(c['new'])[:120]}")
        print(f"    predicted {edit.get('prediction')}  measured train {_brief(c_train)} reg {_brief(c_reg)}"
              f"  => {'ACCEPT' if decision['accepted'] else 'REJECT'} ({decision['reason']})")
        log.append(ev)
        if decision["accepted"]:
            parent, p_train, p_reg, streak, rejected = harness.latest(lineage), c_train, c_reg, 0, []
        else:
            rejected.append({"changes": edit.get("changes"), "reason": decision["reason"]})
            streak += 1
            if streak >= max_rejects:
                print(f"[{body}] plateau: {streak} rejects in a row")
                break
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("body")
    ap.add_argument("--gens", type=int, default=4)
    ap.add_argument("--mtype", default="single")
    ap.add_argument("--difficulty", default="medium")
    a = ap.parse_args()
    run(a.body, a.gens, mtype=a.mtype, difficulty=a.difficulty)
