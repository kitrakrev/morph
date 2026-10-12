"""Multimodal failure memory: each failed trial's final replay frame + its failure text, embedded together
with voyage-multimodal-3.5 (Atlas Embedding API), stored in `failures_mm` with a vector index.

Lets the evolution agent (or a person) find failures that LOOK like a given trajectory.
Isolated add-on: nothing in the gate or the reported numbers depends on it.
"""
from __future__ import annotations

import base64
import io
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import requests  # noqa: E402
from dotenv import dotenv_values  # noqa: E402

from engine.store import db  # noqa: E402
from sim import tier2  # noqa: E402
from sim.missions import make_mission  # noqa: E402
from sim.render import CMAP, _ground_image  # noqa: E402
from sim.world import T  # noqa: E402

URL = "https://ai.mongodb.com/v1/multimodalembeddings"
MODEL = "voyage-multimodal-3.5"
INDEX = "failures_mm_vec"


def _key() -> str:
    return dotenv_values(".env")["VOYAGE_API_KEY"]


def mission_from_id(mission_id: str) -> dict:
    parts = mission_id.split("@")[0].split("-")
    split, body_parts = parts[0], parts[1:]
    seed_i = next(i for i, p in enumerate(body_parts) if p.isdigit())
    body, seed = "-".join(body_parts[:seed_i]), int(body_parts[seed_i])
    rest = body_parts[seed_i + 1:]
    diff, mtype = (rest[0], rest[1]) if len(rest) >= 2 else ("medium", "single")
    return make_mission(body, seed, split, difficulty=diff, mtype=mtype)


def final_frame_png(mission: dict, program: list[dict]) -> bytes:
    """Top-down final state: map, full trajectory, end marker (X if the robot died)."""
    sim = tier2.simulate(mission, program)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=96)
    ax.imshow(_ground_image(sim), cmap=CMAP, vmin=0, vmax=len(T) - 1, origin="lower", interpolation="nearest")
    rid = mission["robots"][0]["id"]
    xs = [f["robots"][rid]["pos"][0] for f in sim.frames]
    ys = [f["robots"][rid]["pos"][1] if len(f["robots"][rid]["pos"]) > 1 else 0 for f in sim.frames]
    alive = sim.frames[-1]["robots"][rid]["alive"]
    ax.plot(xs, ys, "-", color="#d32f2f", lw=2)
    ax.plot(xs[-1], ys[-1], "X" if not alive else "o", color="k" if not alive else "#d32f2f", ms=12)
    ax.set_xticks([])
    ax.set_yticks([])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def embed(text: str | None, png: bytes | None = None, input_type: str | None = None) -> list[float]:
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if png:
        content.append({"type": "image_base64",
                        "image_base64": "data:image/png;base64," + base64.b64encode(png).decode()})
    r = requests.post(URL, timeout=90, headers={"Authorization": f"Bearer {_key()}"},
                      json={"inputs": [{"content": content}], "model": MODEL,
                            **({"input_type": input_type} if input_type else {})})
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def ensure_index():
    if "failures_mm" not in db().list_collection_names():
        db().create_collection("failures_mm")
    coll = db().failures_mm
    if INDEX not in {i["name"] for i in coll.list_search_indexes()}:
        coll.create_search_index({"name": INDEX, "type": "vectorSearch", "definition": {"fields": [
            {"type": "vector", "path": "embedding", "numDimensions": 1024, "similarity": "cosine"},
            {"type": "filter", "path": "body"}]}})


def ingest(body: str, limit: int = 20) -> int:
    """Embed the latest failed train/regression trials of a body (held-out never enters memory)."""
    ensure_index()
    seen, n = set(), 0
    for t in db().trials.find({"body": body, "split": {"$ne": "heldout"}, "score.success": False}).sort("ts", -1):
        key = (t["mission_id"], t["harness_version"])
        if key in seen or db().failures_mm.find_one({"mission_id": t["mission_id"], "version": t["harness_version"]}):
            continue
        seen.add(key)
        png = final_frame_png(mission_from_id(t["mission_id"]), t["program"])
        text = f"{body} v{t['harness_version']}: {t['diagnosis'].split(', sequence')[0]}"
        db().failures_mm.insert_one({"body": body, "version": t["harness_version"], "mission_id": t["mission_id"],
                                     "summary": text, "png": png, "embedding": embed(text, png), "ts": time.time()})
        n += 1
        if n >= limit:
            break
    return n


def similar_by_image(png: bytes, body: str, k: int = 3, exclude_mission: str | None = None) -> list[dict]:
    """Failures whose final trajectory LOOKS like this frame (image-only query)."""
    q = embed(None, png)
    pipe = [{"$vectorSearch": {"index": INDEX, "path": "embedding", "queryVector": q, "numCandidates": 100,
                               "limit": k + 1, "filter": {"body": body}}},
            {"$project": {"_id": 0, "summary": 1, "mission_id": 1, "version": 1,
                          "score": {"$meta": "vectorSearchScore"}}}]
    return [d for d in db().failures_mm.aggregate(pipe) if d["mission_id"] != exclude_mission][:k]
