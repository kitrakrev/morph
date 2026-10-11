"""Render real MuJoCo 3D replays. Usage: python scripts/render_mujoco.py [body ...]

Runs a reference program (charge if needed, go to item, pick, go to target, drop)
for a held-out mission of each body on MuJoCo physics and writes out/mujoco-<body>.mp4.
"""
import sys
import time

from sim import tier2, verifier
from sim.mission import build_sim
from sim.missions import BODIES, mission_set
from sim.render3d import render3d


def program(m: dict) -> list[dict]:
    rid = m["robots"][0]["id"]
    sim = build_sim({**m, "physics": "grid"})
    body = sim.robots[rid].body
    air = "flight" in body.modes or "fixed_wing" in body.modes
    chargers = sim.world.chargers

    def go(p):
        if air:
            return {"robot": rid, "skill": "fly_to", "args": {"pos": list(p), "alt": 4, "avoid_nofly": True}}
        return {"robot": rid, "skill": "move_to", "args": {"pos": list(p)}}

    def near(p):
        return min(chargers, key=lambda c: sum(abs(a - b) for a, b in zip(c, p)))
    out = []
    for g in m["goal"].get("deliveries") or [m["goal"]["deliver"]]:
        it = next(i for i in m["items"] if i["id"] == g["item"])
        if body.kind in ("drone", "hybrid"):
            out += [go(near(it["pos"])), {"robot": rid, "skill": "charge", "args": {}}]
        out += [go(it["pos"]), {"robot": rid, "skill": "pick", "args": {"item": g["item"]}}]
        if body.kind in ("drone", "hybrid"):
            out += [go(near(g["to"])), {"robot": rid, "skill": "charge", "args": {}}]
        out += [go(g["to"]), {"robot": rid, "skill": "drop", "args": {"item": g["item"]}}]
    return out


if __name__ == "__main__":
    for body in sys.argv[1:] or list(BODIES):
        m = mission_set(body, "heldout", 1)[0]
        sim = tier2.simulate({**m, "physics": "mujoco"}, program(m))
        s = verifier.score(m, sim)
        verdict = "DELIVERED" if s["success"] else ("delivered, " if s["delivered"] else "FAILED, ") + \
            ", ".join(k for k, v in s["safety_violations"].items() if v) or "FAILED"
        t0 = time.time()
        out = render3d(sim, f"out/mujoco-{body}.mp4", f"{body} | {m['id']} | {verdict} | {s['energy_wh']} Wh")
        print(out, verdict, f"{len(sim.trace)} poses, {time.time() - t0:.0f}s render")
