"""Promotion gate. Fixed code. A candidate is promoted only if every rule passes.

Held-out missions are never used here; they are scored after acceptance, for reporting only.
"""
from __future__ import annotations

import time

from engine.harness import validate
from engine.store import client, db


def decide(parent: dict, cand: dict, p_train: dict, c_train: dict, p_reg: dict, c_reg: dict) -> dict:
    checks = {}
    errs = sum(t["score"]["skill_errors"] > 0 and not t["score"]["delivered"] for t in c_train["trials"])
    checks["valid_eval"] = errs <= 0.1 * c_train["n"] or c_train["success"] > p_train["success"]
    checks["legal_edit"] = not validate(cand)
    checks["train_improves"] = c_train["success"] > p_train["success"]
    checks["safety_not_worse"] = (c_train["violations"] + c_reg["violations"]) <= (p_train["violations"] + p_reg["violations"])
    checks["no_regression"] = c_reg["success"] >= p_reg["success"]
    gain = max(0, c_train["success"] - p_train["success"])
    # cost = tool calls per successful delivery (a safety stop that saves a mission is not waste)
    cost_p = sum(len(t["program"]) for t in p_train["trials"]) / max(1, p_train["success"])
    cost_c = sum(len(t["program"]) for t in c_train["trials"]) / max(1, c_train["success"])
    checks["cost_ok"] = cost_c <= cost_p * (1.10 + 0.10 * gain)
    accepted = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    return {"accepted": accepted, "checks": checks,
            "reason": "all checks passed" if accepted else "failed: " + ", ".join(failed),
            "metrics": {"parent": {"train": p_train["success"], "reg": p_reg["success"],
                                   "viol": p_train["violations"] + p_reg["violations"], "calls_per_success": round(cost_p, 1)},
                        "candidate": {"train": c_train["success"], "reg": c_reg["success"],
                                      "viol": c_train["violations"] + c_reg["violations"], "calls_per_success": round(cost_c, 1)}}}


def commit(cand: dict, decision: dict, edit: dict) -> dict:
    """Write the harness verdict and the evaluation record atomically."""
    cand = {k: v for k, v in cand.items() if k != "_id"}
    cand["verdict"] = "accepted" if decision["accepted"] else "rejected"
    cand["prediction"] = edit.get("prediction")
    cand["rationale"] = edit.get("rationale")
    ev = {"candidate_id": f"{cand['body_type']}-v{cand['version']}", "body": cand["body_type"],
          "version": cand["version"], "parent": cand["parent"], "edit": edit.get("changes"), "diff": cand.get("diff"),
          "prediction": edit.get("prediction"), "measured": {"train_success_delta":
              decision["metrics"]["candidate"]["train"] - decision["metrics"]["parent"]["train"]},
          "retrieved_failures": edit.get("retrieved", []), "model": edit.get("model"), **decision,
          "ts": time.time()}
    with client().start_session() as s:
        with s.start_transaction():
            db().harnesses.replace_one({"body_type": cand["body_type"], "version": cand["version"]}, cand,
                                       upsert=True, session=s)
            db().evaluations.replace_one({"candidate_id": ev["candidate_id"]}, ev, upsert=True, session=s)
    return ev
