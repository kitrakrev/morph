"""Multimodal failure memory (voyage-multimodal-3.5 via the Atlas Embedding API).

Two stores, one model:
  * `failures` documents carry `mm_embedding` (top-down map picture + diagnosis, written by the runner);
    evolution retrieval fuses it with the text-vector and keyword pipelines in one $rankFusion
    (index `failures_mm` on `failures`; see engine/evolve.similar_failures).
  * `failures_mm` collection holds final replay frames for image-only "looks like this" queries
    (index `failures_mm_vec`; ingest / similar_by_image).
Nothing in the gate or the reported numbers depends on either.
"""
from __future__ import annotations

import base64
import io
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
from dotenv import dotenv_values  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from engine.store import db  # noqa: E402
from sim import tier2  # noqa: E402
from sim.missions import make_mission  # noqa: E402
from sim.render import CMAP, _ground_image  # noqa: E402
from sim.world import T  # noqa: E402

URL = "https://ai.mongodb.com/v1/multimodalembeddings"
ENDPOINTS = (URL, "https://api.voyageai.com/v1/multimodalembeddings")
MODEL = os.environ.get("MORPH_MM_MODEL", "voyage-multimodal-3.5")
INDEX = "failures_mm_vec"
DIMS = 1024
PX = 8  # pixels per cell in failure_image


def _key() -> str | None:
    return os.environ.get("VOYAGE_API_KEY") or dotenv_values(".env").get("VOYAGE_API_KEY")


def enabled() -> bool:
    return bool(_key()) and os.environ.get("MORPH_MM", "1") != "0"


def embed(text: str | None = None, png: bytes | None = None, input_type: str | None = None) -> list[float]:
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if png:
        content.append({"type": "image_base64",
                        "image_base64": "data:image/png;base64," + base64.b64encode(png).decode()})
    body = {"inputs": [{"content": content}], "model": MODEL, **({"input_type": input_type} if input_type else {})}
    last = None
    for url in ENDPOINTS:  # Atlas-issued keys use ai.mongodb.com; Voyage-issued keys use api.voyageai.com
        r = requests.post(url, timeout=90, headers={"Authorization": f"Bearer {_key()}"}, json=body)
        if r.status_code == 200:
            return r.json()["data"][0]["embedding"]
        last = f"{r.status_code} {r.text[:120]}"
    raise RuntimeError(f"multimodal embedding failed: {last}")

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


# ---- map picture for the runner's per-failure embedding ----------------------------------------
COLORS = {T.FLOOR: (200, 200, 195), T.CHARGER: (40, 190, 90), T.RUBBLE: (150, 115, 80), T.STAIRS: (215, 205, 180),
          T.WATER: (60, 120, 200), T.RAIL: (110, 105, 100), T.WALL: (70, 72, 80), T.AIR: (200, 200, 195),
          T.NOFLY: (200, 200, 195)}


def failure_image(sim) -> bytes:
    """Top-down PNG: terrain, building heights (darker = taller), no-fly (red), route, start, failure marks."""
    w = sim.world
    g = w.grid if w.dims == 2 else (w.grid[:, None] if w.dims == 1 else w.grid[:, :, 0])
    W_, H_ = g.shape
    im = Image.new("RGB", (W_ * PX, H_ * PX))
    d = ImageDraw.Draw(im, "RGBA")
    heights = (w.grid == T.WALL).sum(axis=2) if w.dims == 3 else None
    for x in range(W_):
        for y in range(H_):
            c = COLORS.get(T(int(g[x, y])), (200, 200, 195))
            if heights is not None and heights[x, y]:
                c = tuple(max(0, v - 8 * int(heights[x, y])) for v in c)
            d.rectangle([x * PX, y * PX, (x + 1) * PX - 1, (y + 1) * PX - 1], fill=c)
    if w.dims == 3:
        for x, y in np.argwhere((w.grid[:, :, 1:] == T.NOFLY).any(axis=2)):
            d.rectangle([x * PX, y * PX, (x + 1) * PX - 1, (y + 1) * PX - 1], fill=(230, 40, 30, 110))
    for rid in sim.robots:
        pts = [f["robots"][rid]["pos"] for f in sim.frames if rid in f["robots"]]
        xy = [((p[0] + .5) * PX, ((p[1] if len(p) > 1 else 0) + .5) * PX) for p in pts]
        if len(xy) > 1:
            d.line(xy, fill=(255, 140, 0), width=3)
        if xy:
            d.ellipse([xy[0][0] - 5, xy[0][1] - 5, xy[0][0] + 5, xy[0][1] + 5], fill=(30, 120, 255))
    for e in sim.events:
        if e["kind"] in ("battery_death", "collision", "nofly", "skill_error") and e.get("pos"):
            x, y = (e["pos"][0] + .5) * PX, ((e["pos"][1] if len(e["pos"]) > 1 else 0) + .5) * PX
            d.line([x - 6, y - 6, x + 6, y + 6], fill=(255, 0, 0), width=3)
            d.line([x - 6, y + 6, x + 6, y - 6], fill=(255, 0, 0), width=3)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def thumbnail_b64(png: bytes, size: int = 160) -> str:
    im = Image.open(io.BytesIO(png))
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()
