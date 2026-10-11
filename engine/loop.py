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


def run(body: str, gens: int, n_train: int = 6, n_reg: int = 4, max_rejects: int = 3) -> list[dict]:
    train, reg = mission_set(body, "train", n_train), mission_set(body, "regression", n_reg)
    parent = harness.latest(body) or harness.save(harness.bootstrap(BODIES[body]))
    p_train, p_reg = evaluate(parent, train), evaluate(parent, reg)
    print(f"[{body}] v{parent['version']} train {_brief(p_train)} reg {_brief(p_reg)}")
    log, rejected, streak = [], [], 0
    next_version = (db().harnesses.find_one({"body_type": body}, sort=[("version", -1)]) or parent)["version"] + 1
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
            parent, p_train, p_reg, streak, rejected = harness.latest(body), c_train, c_reg, 0, []
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
    a = ap.parse_args()
    run(a.body, a.gens)
