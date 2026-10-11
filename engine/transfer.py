"""Cross-body transfer through taxonomic memory.

When a lineage gets a promoted harness, its lessons are stored under the body's CATEGORY
(taxonomic memory, shared across bodies) and as a procedure (procedural memory).
A new body is warm-started from the best harness of a donor body in the same category:
it inherits the learned workflow, rules, budgets and retrieval policy, but keeps its own
body-specific constraints, tool specs and beliefs.
"""
from __future__ import annotations

import copy

from engine import harness
from engine.memory import Memory
from engine.store import db
from sim.physics import Body

CATEGORY = {"legs": "legged", "wheels": "wheeled", "tracks": "wheeled", "rotors": "aerial", "rail": "rail",
            "fixed_wing": "aerial"}
INHERIT = ("workflow", "rules", "budgets", "retrieval_policy", "reports_read", "skills")


def category(structure: dict) -> str:
    loco = next(iter(structure.get("locomotion", {})), "")
    return CATEGORY.get(loco, "unknown")


def record_lessons(lineage: str, structure: dict) -> dict | None:
    """Store the lineage's best accepted harness as a taxonomic + procedural lesson."""
    best = harness.latest(lineage)
    if not best or best["version"] == 1:
        return None
    cat, kind = category(structure), Body.from_structure(structure).kind
    Memory.save_taxonomic("bodies", kind, f"{kind} is a {cat} body. Best harness: {lineage} v{best['version']}. "
                                          f"Learned rules: {' '.join(best.get('rules', []))}")
    db().taxonomic.update_one({"domain": "bodies", "term": kind},
                              {"$set": {"category": cat, "lineage": lineage, "version": best["version"]}})
    Memory(cat).save_procedure(f"{lineage}-v{best['version']}", f"lessons learned on {kind}",
                               " | ".join(best.get("rules", [])) + f" | workflow {best['workflow']['nodes']}")
    return {"category": cat, "lineage": lineage, "version": best["version"]}


def warm_bootstrap(structure: dict, lineage: str) -> tuple[dict, dict | None]:
    """Bootstrap from structure, then inherit learned fields from the best same-category donor."""
    base = harness.bootstrap(structure)
    base["body_type"] = lineage
    cat = category(structure)
    donor_entry = db().taxonomic.find_one({"domain": "bodies", "category": cat}, sort=[("ts", -1)])
    if not donor_entry:
        return base, None
    donor = db().harnesses.find_one({"body_type": donor_entry["lineage"], "version": donor_entry["version"]})
    for f in INHERIT:
        base[f] = copy.deepcopy(donor[f])
    base["transferred_from"] = {"lineage": donor_entry["lineage"], "version": donor_entry["version"],
                                "category": cat, "fields": list(INHERIT)}
    base["verdict"] = "bootstrap"
    return base, base["transferred_from"]
