"""Multimodal failure memory: every failed trial is embedded as (top-down map picture + diagnosis text)
with Voyage `voyage-multimodal-3.5` (served through MongoDB Atlas) and stored on the failure document.

The picture carries what text loses: where the route went, how close it passed a no-fly zone or a
building block, where the battery died. Evolution retrieval queries this space with text (cross-modal)
and fuses it with the text-vector and keyword pipelines in one `$rankFusion`.
"""
from __future__ import annotations

import base64
import io
import os

import numpy as np
import requests
from PIL import Image, ImageDraw

from sim.world import T

MODEL = os.environ.get("MORPH_MM_MODEL", "voyage-multimodal-3.5")
ENDPOINTS = ("https://ai.mongodb.com/v1/multimodalembeddings", "https://api.voyageai.com/v1/multimodalembeddings")
DIMS = 1024
PX = 8  # pixels per cell

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


def embed(text: str | None = None, png: bytes | None = None, input_type: str = "document") -> list[float]:
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if png:
        content.append({"type": "image_base64", "image_base64": "data:image/png;base64," + base64.b64encode(png).decode()})
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise RuntimeError("VOYAGE_API_KEY not set")
    last = None
    for url in ENDPOINTS:  # Atlas-issued keys use ai.mongodb.com; Voyage-issued keys use api.voyageai.com
        r = requests.post(url, headers={"Authorization": f"Bearer {key}"}, timeout=30,
                          json={"model": MODEL, "input_type": input_type, "inputs": [{"content": content}]})
        if r.status_code == 200:
            return r.json()["data"][0]["embedding"]
        last = f"{r.status_code} {r.text[:120]}"
    raise RuntimeError(f"multimodal embedding failed: {last}")


def thumbnail_b64(png: bytes, size: int = 160) -> str:
    im = Image.open(io.BytesIO(png))
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def enabled() -> bool:
    return bool(os.environ.get("VOYAGE_API_KEY")) and os.environ.get("MORPH_MM", "1") != "0"
