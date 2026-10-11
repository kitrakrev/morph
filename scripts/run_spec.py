"""Run any mission spec on MuJoCo with a reference plan template and render it.

    python scripts/run_spec.py specs/go2-plant-inspection.json [--split heldout] [--n 3] [--video]

The reference template is what a bootstrap planner would write for the goal's task types (no LLM
needed); `python -m engine.loop --spec <file>` evolves a real harness on the same spec.
"""
import argparse

from engine.planner import compile_program
from sim import spec, tier2, verifier
from sim.goals import tasks
from sim.missions import mission_set
from sim.physics import Body


def reference_template(s: dict) -> list[dict]:
    body = Body.from_structure(s["robot"])
    air = not any(m not in ("flight", "fixed_wing") for m in body.modes)

    def go(p):
        return {"skill": "fly_to", "args": {"pos": p, "alt": 6, "avoid_nofly": True}} if air else \
            {"skill": "move_to", "args": {"pos": p}}
    types = [t["type"] for t in tasks(s["goal"])]
    tpl = []
    if "deliver" in types:
        tpl += [go("$ITEM_POS"), {"skill": "pick", "args": {"item": "$ITEM"}}, go("$DEST"),
                {"skill": "drop", "args": {"item": "$ITEM"}}]
    if any(t in types for t in ("visit", "patrol")):
        tpl.append({"foreach": "$TARGETS", "do": [go("$TARGET")]})
    if any(t in types for t in ("inspect", "survey")):
        tpl.append({"foreach": "$TARGETS", "do": [go("$TARGET"), {"skill": "scan", "args": {}}]})
    if "return_home" in types:
        tpl.append(go("$HOME"))
    return tpl


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--video", action="store_true")
    a = ap.parse_args()
    s = spec.load(a.spec)
    tpl = reference_template(s)
    print(s["name"], "| robot", s["robot"]["locomotion"], f"{s['robot']['mass_kg']}kg",
          "| derived from", s["robot"].get("derived_from", {}).get("name", "structure"))
    for m in mission_set(s["name"], a.split, a.n):
        sim = tier2.simulate({**m, "physics": "mujoco"}, compile_program(tpl, m, {"tool_specs": {}}))
        sc = verifier.score(m, sim)
        status = " ".join(f"{t['type']}:{'ok' if t['done'] else 'MISS'}" for t in sc["task_status"])
        print(f"  {m['id']:32s} {'SUCCESS' if sc['success'] else 'fail   '} {status} | "
              f"{sc['energy_wh']}Wh t={sc['time']} viol={sc['violations_total']}")
        if a.video:
            from sim.render3d import render3d
            out = render3d(sim, f"out/{m['id']}.mp4", f"{s['name']} | {m['id']} | "
                                                       f"{'GOAL MET' if sc['success'] else 'FAILED'} | {status}")
            print("   video", out)
