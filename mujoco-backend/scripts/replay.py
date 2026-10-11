"""Render MP4 replays of stored trials. Usage: python scripts/replay.py drone train-drone-104 1 8"""
import sys
from engine.store import db
from sim.missions import make_mission
from sim import tier2
from sim.render import render

body, mission_id, *versions = sys.argv[1:]
split, _, seed = mission_id.rsplit("-", 2)[0], None, int(mission_id.rsplit("-", 1)[1])
mission = make_mission(body, seed, mission_id.split("-")[0])
for v in versions:
    t = db().trials.find_one({"body": body, "mission_id": mission_id, "harness_version": int(v)}, sort=[("ts", -1)])
    if not t:
        print("no trial for v", v); continue
    sim = tier2.simulate(mission, t["program"])
    s = t["score"]
    label = "DELIVERED" if s["success"] else ("FAILED: " + t["diagnosis"].split(",")[0])
    out = f"out/{mission_id}-v{v}.mp4"
    render(sim, out, f"{body} harness v{v} | {mission_id} | {label}")
    print(out, label)
