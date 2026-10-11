"""Harness documents: bootstrap v1 from body structure, validate edits, version in Atlas."""
from __future__ import annotations

import copy
import time

from sim.physics import Body
from sim.skills import PRIMITIVES
from engine.store import db

NODES = ("plan", "tier1_check", "replan", "tier2_sim", "verify")
LOCKED = {"constraints.nofly": "hard", "constraints.collision": "hard"}
EDITABLE = ("workflow", "rules", "constraints", "tool_specs", "reports_read", "retrieval_policy",
            "model_routing", "budgets", "skills", "beliefs")


SUPPORTED_LOCOMOTION = ("wheels", "legs", "rotors", "tracks", "rail")


class UnsupportedBody(ValueError):
    pass


def bootstrap(structure: dict) -> dict:
    """Initial harness derived from structure only: plausible but naive beliefs.

    Refuses bodies outside the known movement vocabulary instead of guessing."""
    loco = set(structure.get("locomotion", {}))
    unknown = loco - set(SUPPORTED_LOCOMOTION)
    if not loco or unknown:
        raise UnsupportedBody(
            f"cannot bootstrap: locomotion {sorted(unknown) or 'missing'} is outside the supported "
            f"vocabulary {list(SUPPORTED_LOCOMOTION)}. A new movement mode needs a mode plugin "
            f"(tier 2, sandboxed + gated) and human approval (tier 3) before this body can be planned for.")
    body = Body.from_structure(structure)
    flying = "flight" in body.modes
    return {
        "version": 1, "parent": None, "body_type": body.kind,
        "workflow": {"nodes": ["plan", "tier2_sim", "verify"],
                     "edges": [["plan", "tier2_sim"], ["tier2_sim", "verify"]]},
        "rules": ["Deliver the item to the destination.", "Take the shortest route."],
        "constraints": {"payload_kg_max": body.payload_max_kg, "nofly": "hard", "collision": "hard",
                        **({"altitude_max": 8} if flying else {})},
        "tool_specs": {"fly_to": {"alt": [1, 8], "avoid_nofly": False}} if flying else {"move_to": {}},
        "beliefs": {"wh_per_cell": {m: {"wheel": 0.15, "legs": 0.35, "rail": 0.05, "flight": 0.4}[m]
                                    for m in body.modes}, "reserve_frac": 0.0},
        "reports_read": ["terrain_summary"],
        "retrieval_policy": {"episodes": 0, "procedures": 0, "same_body_category": True},
        "model_routing": {"plan": "cheap"},
        "budgets": {"replans": 0, "tool_calls": 40, "tier2_runs": 1},
        "skills": [],
        "prediction": None, "verdict": "bootstrap", "diff": None, "created": time.time(),
    }


def get(d: dict, dotted: str):
    for k in dotted.split("."):
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d


def validate(h: dict) -> list[str]:
    """Reasons an (edited) harness is illegal. Empty list means legal."""
    errs = []
    for path, val in LOCKED.items():
        if get(h, path) != val:
            errs.append(f"locked constraint {path} must stay {val!r}")
    for n in h["workflow"]["nodes"]:
        if n not in NODES:
            errs.append(f"unknown workflow node {n}")
    for need in ("plan", "tier2_sim", "verify"):
        if need not in h["workflow"]["nodes"]:
            errs.append(f"workflow must keep {need}")
    for t in h.get("tool_specs", {}):
        if t not in PRIMITIVES:
            errs.append(f"tool {t} is not a primitive")
    for s in h.get("skills", []):
        if not isinstance(s, dict) or any(st.get("skill") not in PRIMITIVES for st in s.get("steps", [])):
            errs.append(f"composed skill {s if not isinstance(s, dict) else s.get('name')} uses non-primitive")
    return errs


def edit_budget(version: int) -> int:
    """Edits allowed per candidate shrink with lineage age: 3 at v1-v2, 2 at v3-v4, then 1."""
    return 3 if version <= 2 else 2 if version <= 4 else 1


def apply_edit(parent: dict, edit: dict) -> dict:
    """edit = {"changes": [{"field": <EDITABLE>, "value": <new value>}, ...]} -> child harness (not saved)."""
    changes = edit.get("changes") or [{"field": edit.get("field"), "value": edit.get("value")}]
    if len(changes) > edit_budget(parent["version"]):
        raise ValueError(f"{len(changes)} changes exceeds edit budget {edit_budget(parent['version'])}")
    child = copy.deepcopy({k: v for k, v in parent.items() if k != "_id"})
    diff = []
    for c in changes:
        if c.get("field") not in EDITABLE:
            raise ValueError(f"field {c.get('field')} not editable")
        diff.append({"field": c["field"], "old": parent.get(c["field"]), "new": c["value"]})
        child[c["field"]] = c["value"]
    child.update(version=parent["version"] + 1, parent=parent["version"], verdict="candidate",
                 diff=diff, created=time.time())
    return child


def latest(body_type: str, accepted_only: bool = True) -> dict | None:
    q = {"body_type": body_type}
    if accepted_only:
        q["verdict"] = {"$in": ["bootstrap", "accepted"]}
    return db().harnesses.find_one(q, sort=[("version", -1)])


def save(h: dict) -> dict:
    h = {k: v for k, v in h.items() if k != "_id"}
    db().harnesses.replace_one({"body_type": h["body_type"], "version": h["version"]}, h, upsert=True)
    return db().harnesses.find_one({"body_type": h["body_type"], "version": h["version"]})
