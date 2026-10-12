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


def _terrain(spec: dict | None) -> dict:
    t = (spec or {}).get("terrain", {})
    return {k: t[k] for k in ("difficulty",) if k in t}


def crosses_nofly(m: dict) -> bool:
    """True if a direct flight at the robot's lowest allowed altitude, without avoidance, enters a no-fly zone."""
    from sim import tier2
    r, it, to = m["robots"][0], m["items"][0], m["goal"]["deliver"]["to"]
    lo = ((r.get("capabilities") or {}).get("fly_to") or {}).get("alt", [1, 8])[0]
    prog = [{"robot": r["id"], "skill": "fly_to", "args": {"pos": it["pos"], "alt": lo}},
            {"robot": r["id"], "skill": "pick", "args": {"item": it["id"]}},
            {"robot": r["id"], "skill": "fly_to", "args": {"pos": to, "alt": lo}}]
    return any(e["kind"] == "nofly" for e in tier2.simulate(m, prog).events)


def train_set(body: str, n: int, difficulty: str, mtype: str, spec: dict | None) -> list[dict]:
    """Train missions. A spec may ask for half of them to require no-fly avoidance (so the problem is visible
    in training, not only on held-out maps). Held-out generation is never touched."""
    base = patch(mission_set(body, "train", n, difficulty, mtype), spec)
    if not (spec or {}).get("training", {}).get("include_nofly_routes"):
        return base
    pool = patch(mission_set(body, "train", 60, difficulty, mtype), spec)
    cross = [m for m in pool if crosses_nofly(m)][: n // 2]
    keep = [m for m in base if m["id"] not in {c["id"] for c in cross}][: n - len(cross)]
    return keep + cross


def patch(missions: list[dict], spec: dict | None) -> list[dict]:
    """Apply a robot spec file: its structure and declared capabilities replace the preset body's."""
    if not spec:
        return missions
    for m in missions:
        r = m["robots"][0]
        r["structure"] = spec["structure"]
        if spec.get("capabilities"):
            r["capabilities"] = spec["capabilities"]
        m["id"] = f"{m['id']}@{spec['id']}"
    return missions


def _brief(e: dict) -> str:
    return f"{e['success']}/{e['n']} (n={e['n']}) viol={e['violations']}"


def adapt_curve(body: str, lineage: str, start: str = "cold", gens: int = 3, target: float = 0.8,
                spec: dict | None = None) -> dict:
    """Generations until held-out success >= target, for a cold or warm (transfer) start."""
    from engine import transfer
    heldout = patch(mission_set(body, "heldout", 12, **_terrain(spec)), spec)
    if not harness.latest(lineage):
        if spec:
            h = harness.bootstrap(spec["structure"], spec.get("capabilities"))
            h["body_type"] = lineage
        elif start == "warm":
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
        run(body, 1, lineage=lineage, spec=spec)
        if harness.latest(lineage)["version"] == before:  # rejected: one more proposal counts as a generation
            continue
    reached = next((c["gen"] for c in curve if c["heldout"] >= target * c["n"]), None)
    return {"lineage": lineage, "start": start, "curve": curve, "gens_to_adapt": reached}


def run(body: str, gens: int, n_train: int = 6, n_reg: int = 4, max_rejects: int = 3,
        mtype: str = "single", difficulty: str = "medium", lineage: str | None = None,
        spec: dict | None = None) -> list[dict]:
    lineage = lineage or body
    if spec:
        difficulty = _terrain(spec).get("difficulty", difficulty)
    train = train_set(body, n_train, difficulty, mtype, spec)
    # regression always guards the original task (medium / single), so a new skill can't break the old one
    reg = patch(mission_set(body, "regression", n_reg), spec)
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
    ap.add_argument("--lineage", help="store versions under this lineage name (default: the body name)")
    ap.add_argument("--max-rejects", type=int, default=3, help="stop after this many consecutive rejections")
    a = ap.parse_args()
    run(a.body, a.gens, mtype=a.mtype, difficulty=a.difficulty, lineage=a.lineage, max_rejects=a.max_rejects)
