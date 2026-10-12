"""Workflow-graph runner: executes a harness's workflow per mission, checkpoints to Atlas,
resumes after a kill, and writes trials, failures and episodic memory."""
from __future__ import annotations

import time
import uuid

from sim import tier1, tier2, verifier
from sim.mission import build_sim
from engine.memory import Memory
from engine.planner import compile_program, plan_template
from engine.store import db
from engine import telemetry


def sim_program(sim) -> list[dict]:
    return [e for e in sim.events if e["kind"] == "call"]


def _diagnose(score: dict, t1: dict | None, sim) -> str:
    parts = []
    v = score["safety_violations"]
    if v["battery_death"]:
        dead = next(e for e in sim.events if e["kind"] == "battery_death")
        parts.append(f"battery died at {dead['pos']}{' mid-air' if dead.get('airborne') else ''}")
    if v["nofly"]:
        parts.append(f"entered no-fly zone {v['nofly']} cells")
    if v["payload"]:
        parts.append("payload over limit")
    if score["reserve_breach"]:
        parts.append("finished below battery reserve")
    if not score["deadline_met"]:
        parts.append(f"missed deadline ({score['time']} steps)")
    if not score["delivered"] and not parts:
        errs = [e["reason"] for e in sim.events if e["kind"] == "skill_error"]
        parts.append("not delivered: " + "; ".join(errs[:2]) if errs else "not delivered")
    if t1 and not t1["pass"]:
        parts.append("tier1: " + "; ".join(t1["reasons"][:2]))
    moved = max(1, sum(1 for a, b in zip(sim.frames, sim.frames[1:])
                       for rid in a["robots"] if a["robots"][rid]["pos"] != b["robots"][rid]["pos"]))
    parts.append(f"energy {score['energy_wh']}Wh over {moved} cells = measured {score['energy_wh'] / moved:.2f} Wh/cell")
    parts.append("sequence " + " > ".join(f"{c['skill']}{'' if c['ok'] else '(failed)'}" for c in sim_program(sim)))
    if t1:
        parts.append(f"tier1 estimated {sum(t1['est_wh'].values()):.1f}Wh")
    return ", ".join(parts)


def run_mission(h: dict, template: dict, mission: dict, run_id: str | None = None) -> dict:
    """Execute one mission through the harness workflow. Resumable by run_id."""
    run_id = run_id or f"{h['body_type']}-v{h['version']}-{mission['id']}-{uuid.uuid4().hex[:6]}"
    cp = db().checkpoints.find_one({"run_id": run_id}) or {"run_id": run_id, "done": [], "state": {}}
    st = cp["state"]
    nodes = h["workflow"]["nodes"]

    def save(node):
        cp["done"].append(node)
        db().checkpoints.replace_one({"run_id": run_id}, {**cp, "state": st, "ts": time.time()}, upsert=True)

    if "plan" not in cp["done"]:
        st["program"] = compile_program(template["primary"], mission, h)
        st["which"] = "primary"
        save("plan")
    if "tier1_check" in nodes and "tier1_check" not in cp["done"]:
        st["t1"] = tier1.check(mission, st["program"], {**h.get("beliefs", {}),
                                                         "payload_kg_max": h["constraints"].get("payload_kg_max")})
        save("tier1_check")
    replans = h.get("budgets", {}).get("replans", 0)
    if "replan" in nodes and "replan" not in cp["done"] and st.get("t1") and not st["t1"]["pass"] and replans > 0:
        st["program"] = compile_program(template["fallback"], mission, h)
        st["which"] = "fallback"
        st["t1"] = tier1.check(mission, st["program"], {**h.get("beliefs", {}),
                                                         "payload_kg_max": h["constraints"].get("payload_kg_max")})
        save("replan")
    sim = tier2.simulate(mission, st["program"])
    score = verifier.score(mission, sim)
    diagnosis = "" if score["success"] else _diagnose(score, st.get("t1"), sim)
    trial = {"run_id": run_id, "harness_version": h["version"], "harness_id": str(h.get("_id")),
             "body": h["body_type"], "mission_id": mission["id"], "split": mission["split"],
             "program": st["program"], "which": st.get("which"), "tier1": st.get("t1"), "score": score,
             "diagnosis": diagnosis, "ts": time.time()}
    db().trials.insert_one(dict(trial))
    db().metrics.insert_one({"ts": __import__("datetime").datetime.utcnow(),
                             "meta": {"body": h["body_type"], "version": h["version"], "split": mission["split"]},
                             "success": int(score["success"]), "violations": score["violations_total"],
                             "energy": score["energy_wh"]})
    if mission["split"] != "heldout":  # held-out outcomes never reach agent-readable memory
        if not score["success"]:
            doc = {"body": h["body_type"], "version": h["version"], "mission_id": mission["id"],
                   "summary": f"{h['body_type']} v{h['version']}: {diagnosis}", "ts": time.time()}
            try:  # failure picture + diagnosis -> one multimodal vector (index failures_mm)
                from engine import multimodal
                doc["mm_embedding"] = multimodal.embed(doc["summary"],
                                                       multimodal.final_frame_png(mission, st["program"]))
            except Exception as e:
                print(f"    multimodal embedding skipped ({str(e)[:60]})")
            db().failures.insert_one(doc)
        Memory(h["body_type"]).save_episode(
            title=f"{mission['id']} v{h['version']}",
            content=f"{'SUCCESS' if score['success'] else 'FAIL'} {h['body_type']} carrying "
                    f"{mission['goal']['deliver']['weight_kg']}kg: {diagnosis or 'delivered safely'}",
            version=h["version"], mission_id=mission["id"])
    telemetry.trial_event(trial, model=template.get("model"))
    db().checkpoints.delete_one({"run_id": run_id})
    trial["_sim"] = sim
    return trial


def evaluate(h: dict, missions: list[dict]) -> dict:
    body = build_sim(missions[0]).robots[missions[0]["robots"][0]["id"]].body
    template = plan_template(h, {"kind": body.kind, "modes": list(body.modes), "mass_kg": body.mass_kg,
                                 "battery_wh": body.battery_wh, "payload_max_kg": body.payload_max_kg},
                             build_sim(missions[0]).world.summary())
    trials = [run_mission(h, template, m) for m in missions]
    n = len(trials)
    return {"template": template, "trials": trials, "n": n,
            "success": sum(t["score"]["success"] for t in trials),
            "violations": sum(t["score"]["violations_total"] for t in trials),
            "energy": round(sum(t["score"]["energy_wh"] for t in trials), 1)}
