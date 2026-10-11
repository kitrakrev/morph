"""Every harness attempt on one mission, replayed in MuJoCo 3D, with the inputs and the CURRENT harness
parameters beside it (changed fields marked ● in green), result and gate verdict. Same layout as
scripts/attempts_with_params.py, but the left side is the real-dynamics 3D replay plus a live top-down
minimap and altitude strip.

    PYTHONPATH=. uv run python scripts/attempts_with_params_3d.py [mission_id] [body]
"""
import json
import subprocess
import sys
import textwrap

import numpy as np
from PIL import Image, ImageDraw

from engine.multimodal import COLORS
from engine.store import db
from sim import tier2, verifier
from sim.goals import tasks
from sim.missions import BODIES, make_mission
from sim.render3d import _font, render3d
from sim.world import NOFLY_CEIL, T

MID = sys.argv[1] if len(sys.argv) > 1 else "train-drone-100"
BODY = sys.argv[2] if len(sys.argv) > 2 else "drone"
CW, CH = 1920, 1080
BG, INK, MUTED, NEW, OK, BAD = (251, 251, 248), (29, 36, 32), (107, 115, 110), (0, 120, 76), (46, 125, 50), (198, 40, 40)
FIELDS = ["workflow", "rules", "tool_specs", "beliefs", "budgets", "retrieval_policy"]
F_TITLE, F_H, F_B, F_S = _font(30), _font(24), _font(17), _font(15)

p = MID.split("-")
mission = make_mission(BODY, int(p[2]), p[0], difficulty=p[3] if len(p) > 4 else "medium",
                       mtype=p[4] if len(p) > 4 else "single")
first = db().llm_cache.find_one({"tier": "claude"}, sort=[("ts", 1)])
cut = (first["ts"] - 60) if first else 0
latest = {}
for t in db().trials.find({"body": BODY, "mission_id": MID, "ts": {"$gte": cut}}).sort("ts", 1):
    latest[t["harness_version"]] = t
versions = sorted(latest)


def fmt(field, val):
    if field == "workflow":
        return " → ".join(val["nodes"])
    if field == "rules":
        return " | ".join(val)
    return json.dumps(val, separators=(", ", ": "))


def minimap(sim, trace, k, size=300):
    w = sim.world
    g = w.grid if w.dims == 2 else w.grid[:, :, 0]
    px = size / max(g.shape)
    im = Image.new("RGB", (int(g.shape[0] * px), int(g.shape[1] * px)), BG)
    d = ImageDraw.Draw(im, "RGBA")
    for x in range(g.shape[0]):
        for y in range(g.shape[1]):
            yy = g.shape[1] - 1 - y  # north up
            d.rectangle([x * px, yy * px, (x + 1) * px, (yy + 1) * px], fill=COLORS.get(T(int(g[x, y])), (200, 200, 195)))
    if w.dims == 3:
        for x, y in np.argwhere((w.grid[:, :, 1:] == T.NOFLY).any(axis=2)):
            yy = g.shape[1] - 1 - y
            d.rectangle([x * px, yy * px, (x + 1) * px, (yy + 1) * px], fill=(230, 60, 50, 150))
    to_px = lambda q: (q[0] * px, (g.shape[1] - q[1]) * px)  # noqa: E731  metres -> pixels
    for task in tasks(mission["goal"]):
        if "to" in task:
            c = to_px((task["to"][0] + .5, task["to"][1] + .5))
            d.regular_polygon((c, 9), 5, fill=(250, 200, 30), outline=INK)
    pts = [to_px(f["pos"]) for f in trace[:k + 1]]
    if len(pts) > 1:
        d.line(pts, fill=(211, 47, 47), width=3)
    if pts:
        d.ellipse([pts[-1][0] - 5, pts[-1][1] - 5, pts[-1][0] + 5, pts[-1][1] + 5], fill=(211, 47, 47), outline=INK)
    return im


def altitude(trace, k, w=800, h=300):
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im, "RGBA")
    x0, y0, x1, y1 = 60, 20, w - 20, h - 40
    zmax, tmax = 10, max(trace[-1]["t"], 1)
    zy = lambda z: y1 - (y1 - y0) * min(z, zmax) / zmax  # noqa: E731
    d.rectangle([x0, zy(NOFLY_CEIL), x1, zy(1)], fill=(229, 115, 115, 50))
    d.text((x0 + 8, zy(NOFLY_CEIL) + 4), f"no-fly band z<{NOFLY_CEIL} (where zone active)", font=F_S, fill=BAD)
    d.rectangle([x0, y0, x1, y1], outline=MUTED)
    for z in range(0, zmax + 1, 2):
        d.text((x0 - 30, zy(z) - 8), str(z), font=F_S, fill=MUTED)
    d.text((x0, y1 + 10), "time (s, MuJoCo)", font=F_S, fill=MUTED)
    d.text((8, y0), "alt", font=F_S, fill=MUTED)
    pts = [(x0 + (x1 - x0) * f["t"] / tmax, zy(f["pos"][2] - 0.5 if f["pos"][2] > 0.3 else 0)) for f in trace[:k + 1]]
    if len(pts) > 1:
        d.line(pts, fill=(211, 47, 47), width=3)
    return im


def right_panel(n, v, h, prev, t, score, final):
    im = Image.new("RGB", (680, CH - 70), BG)
    d = ImageDraw.Draw(im)
    y = 10

    def line(txt, color=INK, font=F_B, wrap=62, gap=23):
        nonlocal y
        for part in textwrap.wrap(txt, wrap) or [""]:
            d.text((0, y), part, font=font, fill=color)
            y += gap
    st = mission["robots"][0]["structure"]
    line("INPUTS", MUTED, F_H, gap=32)
    line(f"robot    {BODY} · {', '.join(f'{k} {v}' for k, v in st['locomotion'].items())} · {st['mass_kg']} kg · "
         f"{st['battery_wh']} Wh · payload {st['payload_bay_kg']} kg" + (f" · {st['model']}" if st.get("model") else ""))
    line(f"terrain  {mission['terrain'].get('dims')}D city · {mission['terrain'].get('difficulty')} · "
         f"no-fly below z={NOFLY_CEIL} · wind · chargers")
    goal = mission["goal"]
    line("goal     " + "; ".join(f"{x['type']} {x.get('item', '')}".strip() for x in tasks(goal)) +
         f" · 0 violations · ≥{int(100 * goal.get('reserve_frac', 0))}% reserve")
    line("planner  Claude Haiku (fixed)   evolver  Claude Sonnet   physics  MuJoCo")
    y += 14
    line(f"HARNESS v{v}  ·  attempt {n} of {len(versions)}", INK, F_H, gap=36)
    for f in FIELDS:
        changed = prev is not None and prev.get(f) != h.get(f)
        line(("● " if changed else "  ") + f, NEW if changed else MUTED, F_B, gap=22)
        line("   " + fmt(f, h.get(f)), NEW if changed else INK, F_S, wrap=70, gap=19)
    y += 14
    if final:
        ok = score["success"]
        why = t["diagnosis"].split(", energy")[0].replace("not delivered: ", "").split(";")[0]
        line("RESULT  " + ("DELIVERED" if ok else "FAILED: " + why), OK if ok else BAD, F_H, wrap=40, gap=32)
        gate = {1: "bootstrap (no gate)"}.get(v, "gate ACCEPTED" if h.get("verdict") == "accepted"
                                              else "gate REJECTED → harness discarded")
        line(gate, OK if h.get("verdict") == "accepted" else (MUTED if v == 1 else BAD), F_H, wrap=40, gap=32)
        line(f"MuJoCo: {score['energy_wh']} Wh · {score['time']} steps · violations {score['violations_total']}"
             f"   (recorded grid run: {'success' if t['score']['success'] else 'fail'})", MUTED, F_S, wrap=70)
    else:
        line("running in MuJoCo ...", MUTED, F_H)
    return im


files, prev = [], None
for n, v in enumerate(versions, 1):
    t, h = latest[v], db().harnesses.find_one({"body_type": BODY, "version": v})
    sim = tier2.simulate({**mission, "physics": "mujoco"}, t["program"])
    score = verifier.score(mission, sim)
    trace = [f for f in sim.trace if f["robot"] == mission["robots"][0]["id"]]

    def compose(view, f, k, progress, n=n, v=v, h=h, prev=prev, t=t, score=score, sim=sim, trace=trace):
        canvas = Image.new("RGB", (CW, CH), BG)
        d = ImageDraw.Draw(canvas)
        title = "Same mission, same model — only the harness changes"
        d.text(((CW - d.textlength(title, font=F_TITLE)) / 2, 16), title, font=F_TITLE, fill=INK)
        v3 = Image.fromarray(view).resize((1152, 648))
        canvas.paste(v3, (24, 70))
        d.rectangle([24, 70, 24 + 1152, 70 + 648], outline=MUTED)
        lab = f"attempt {n} · harness v{v}   t={f['t']:.1f}s   battery {int(100 * f['battery'])}%"
        d.rectangle([24, 70, 24 + 20 + d.textlength(lab, font=F_B), 100], fill=(255, 255, 255))
        d.text((34, 75), lab, font=F_B, fill=INK)
        kk = min(range(len(trace)), key=lambda i: abs(trace[i]["t"] - f["t"]))
        canvas.paste(minimap(sim, trace, kk), (24, 740))
        canvas.paste(altitude(trace, kk, 820, 320), (356, 735))
        canvas.paste(right_panel(n, v, h, prev, t, score, progress >= 1.0), (1210, 70))
        return canvas
    out = f"out/attempts3d-p{n}.mp4"
    render3d(sim, out, compose=compose, hud=False, max_seconds=14)
    files.append(out)
    print(out, f"v{v}", "MuJoCo", "OK" if score["success"] else "FAIL", h.get("verdict"))
    prev = h
with open("out/attempts3d-list.txt", "w") as fh:
    fh.writelines(f"file '{f.split('/')[-1]}'\n" for f in files)
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", "attempts3d-list.txt",
                "-c", "copy", "attempts-with-params-3d.mp4"], cwd="out", check=True)
print("out/attempts-with-params-3d.mp4")
