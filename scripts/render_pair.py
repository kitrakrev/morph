"""Side-by-side replay of one mission under two harness versions (the hero clip).

Usage: PYTHONPATH=. uv run python scripts/render_pair.py drone train-drone-105 1 17
"""
import sys

from engine.store import db
from sim import tier2
from sim.missions import make_mission
from sim.render import render_pair


def load(body: str, mission_id: str, version: int):
    seed = int(mission_id.split("-")[2])
    split = mission_id.split("-")[0]
    parts = mission_id.split("-")
    diff, mtype = (parts[3], parts[4]) if len(parts) > 4 else ("medium", "single")
    mission = make_mission(body, seed, split, difficulty=diff, mtype=mtype)
    t = db().trials.find_one({"body": body, "mission_id": mission_id, "harness_version": version}, sort=[("ts", -1)])
    if t is None:
        raise SystemExit(f"no trial for {body} v{version} on {mission_id}")
    sim = tier2.simulate(mission, t["program"])
    s = t["score"]
    result = "DELIVERED safely" if s["success"] else "FAILED: " + t["diagnosis"].split(",")[0]
    return sim, mission, f"harness v{version} | mission {mission_id}", result, s["success"]


if __name__ == "__main__":
    body, mission_id, va, vb = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    out = f"out/pair-{mission_id}-v{va}-v{vb}.mp4"
    render_pair(load(body, mission_id, va), load(body, mission_id, vb), out,
                f"Same mission, same planner model: harness v{va} vs v{vb}")
    print(out)
