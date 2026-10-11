"""Embed existing failures multimodally (map picture + diagnosis) and create the failures_mm index.

    PYTHONPATH=. uv run python scripts/backfill_multimodal.py [--limit N]
"""
import argparse

from engine import multimodal
from engine.store import _search_index, db
from sim import tier2
from sim.missions import BODIES, make_mission


def mission_for(body: str, mid: str):
    p = mid.split("-")
    try:
        seed_i = next(i for i, x in enumerate(p) if x.isdigit())
        diff, mtype = (p[seed_i + 1], p[seed_i + 2]) if len(p) > seed_i + 2 else ("medium", "single")
        return make_mission(body, int(p[seed_i]), p[0], difficulty=diff, mtype=mtype)
    except (StopIteration, KeyError, ValueError):
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    a = ap.parse_args()
    d = db()
    _search_index(d.failures, "failures_mm",
                  {"fields": [{"type": "vector", "path": "mm_embedding", "numDimensions": 1024, "similarity": "cosine"},
                              {"type": "filter", "path": "body"}]}, "vectorSearch")
    done = skipped = 0
    for f in d.failures.find({"mm_embedding": {"$exists": False}}).limit(a.limit):
        body = f["body"]
        t = d.trials.find_one({"body": body, "mission_id": f["mission_id"], "harness_version": f["version"]},
                              sort=[("ts", -1)])
        m = mission_for(body, f["mission_id"]) if body in BODIES else None
        if not t or not m:
            skipped += 1
            continue
        sim = tier2.simulate({**m, "physics": "grid"}, t["program"])  # same physics the trial was scored on
        png = multimodal.failure_image(sim)
        d.failures.update_one({"_id": f["_id"]}, {"$set": {"mm_embedding": multimodal.embed(f["summary"], png),
                                                           "map_png": multimodal.thumbnail_b64(png)}})
        done += 1
    print(f"embedded {done}, skipped {skipped} (no trial or unknown body)")
