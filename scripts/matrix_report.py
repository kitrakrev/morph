"""Print the evaluation matrix as markdown from Atlas trials. Cells never run say 'not run today'.

Usage: PYTHONPATH=. uv run python scripts/matrix_report.py > out/matrix.md
"""
from collections import defaultdict

from engine.store import db
from sim.missions import BODIES, DIFFICULTIES, MISSION_TYPES

best = {h["body_type"]: h["version"] for h in db().harnesses.aggregate([
    {"$match": {"verdict": {"$in": ["bootstrap", "accepted"]}}},
    {"$sort": {"version": -1}}, {"$group": {"_id": "$body_type", "version": {"$first": "$version"}}},
    {"$project": {"body_type": "$_id", "version": 1}}])}
cells = defaultdict(lambda: [0, 0, 0])
for t in db().trials.find({"split": "heldout"}, {"body": 1, "mission_id": 1, "harness_version": 1, "score": 1}):
    if t["harness_version"] != best.get(t["body"]):
        continue
    parts = t["mission_id"].split("-")
    diff, mtype = (parts[3], parts[4]) if len(parts) > 4 else ("medium", "single")
    c = cells[(t["body"], diff, mtype)]
    c[0] += t["score"]["success"]; c[1] += 1; c[2] += t["score"]["violations_total"]

print("| body | difficulty | " + " | ".join(MISSION_TYPES) + " |")
print("|---|---|" + "---|" * len(MISSION_TYPES))
for b in BODIES:
    for d in DIFFICULTIES:
        row = []
        for m in MISSION_TYPES:
            s, n, v = cells.get((b, d, m), (0, 0, 0))
            row.append(f"**{s}/{n}**{' ⚠️' + str(v) if v else ''}" if n else "not run today")
        print(f"| {b}{' (held-out body)' if b == 'hexapod' else ''} | {d} | " + " | ".join(row) + " |")

print("\n| body | best version | edits proposed | accepted | rejected |")
print("|---|---|---|---|---|")
for b in BODIES:
    evs = list(db().evaluations.find({"body": b}, {"accepted": 1}))
    if evs or b in best:
        acc = sum(e["accepted"] for e in evs)
        print(f"| {b} | v{best.get(b, '-')} | {len(evs)} | {acc} | {len(evs) - acc} |")
