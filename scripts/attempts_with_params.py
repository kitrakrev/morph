"""Left: map replay of each attempt on one mission. Right: inputs + the CURRENT harness parameters,
changed fields highlighted, result and gate verdict.
Usage: PYTHONPATH=. uv run python scripts/attempts_with_params.py [body] [mission_id]"""
import json
import subprocess
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

import sim.render as R
from engine.store import db
from sim import tier2
from engine.multimodal import mission_from_id
from sim.missions import BODIES

BODY = sys.argv[1] if len(sys.argv) > 1 else "drone"
MID = sys.argv[2] if len(sys.argv) > 2 else "train-drone-100"
TAG = "" if BODY == "drone" else f"-{BODY}"
R.MAX_FRAMES, R.HOLD_S, R.FPS = 40, 3.0, 6
INK, MUTED, NEW, OK, BAD = "#1d2420", "#6b736e", "#00784C", "#2e7d32", "#c62828"
FIELDS = ["workflow", "rules", "tool_specs", "beliefs", "budgets", "retrieval_policy"]

mission = mission_from_id(MID)
cut = db().llm_cache.find_one({"tier": "claude"}, sort=[("ts", 1)])["ts"] - 60 if BODY == "drone" else 0
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


def right_panel(ax, n, v, h, prev, t):
    ax.axis("off")
    y = 0.98

    def line(txt, color=INK, size=10.5, bold=False, wrap=58, gap=0.032):
        nonlocal y
        for part in textwrap.wrap(txt, wrap) or [""]:
            ax.text(0, y, part.replace("$", r"\$"), color=color, fontsize=size, fontweight="bold" if bold else "normal",
                    family="DejaVu Sans Mono" if not bold else "DejaVu Sans", transform=ax.transAxes, va="top")
            y -= gap

    s = BODIES[BODY]
    loco = ", ".join(s["locomotion"])
    item = mission["goal"]["deliver"]
    line("INPUTS", MUTED, 10, True)
    line(f"robot    {BODY} · {loco} · {s['mass_kg']} kg · {s['battery_wh']} Wh · payload {s['payload_bay_kg']} kg")
    line("terrain  3D city · no-fly below z=6 · wind · chargers" + (" · land on open cells only" if BODY == "fixed_wing_vtol" else ""))
    line(f"goal     deliver {item['item']} ({item['weight_kg']} kg) · 0 violations · ≥20% reserve")
    line("planner  Claude Haiku (fixed)   evolver  Claude Sonnet")
    y -= 0.02
    line(f"HARNESS v{v}  ·  attempt {n} of {len(versions)}", INK, 13, True, gap=0.045)
    for f in FIELDS:
        changed = prev is not None and prev.get(f) != h.get(f)
        line(("● " if changed else "  ") + f, NEW if changed else MUTED, 10, True, gap=0.03)
        line("   " + fmt(f, h.get(f)), NEW if changed else INK, 9.5, wrap=60, gap=0.028)
    y -= 0.02
    ok = t["score"]["success"]
    why = t["diagnosis"].split(", energy")[0].replace("not delivered: ", "").split(";")[0]
    line("RESULT  " + ("DELIVERED" if ok else "FAILED: " + why), OK if ok else BAD, 11, True)
    gate = {1: "bootstrap (no gate)", **({8: "accepted in the earlier run"} if BODY == "drone" else {})}.get(
        v, "gate ACCEPTED" if h.get("verdict") == "accepted" else "gate REJECTED → harness discarded")
    line(gate, OK if (h.get("verdict") == "accepted" and not (BODY == "drone" and v == 8)) else (MUTED if v == 1 or (BODY == "drone" and v == 8) else BAD), 11, True)


files, prev = [], None
for n, v in enumerate(versions, 1):
    t, h = latest[v], db().harnesses.find_one({"body_type": BODY, "version": v})
    sim = tier2.simulate(mission, t["program"])
    fig = plt.figure(figsize=(16, 9), facecolor="#FBFBF8")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], left=0.03, right=0.98, top=0.9, bottom=0.08, wspace=0.05)
    ok = t["score"]["success"]
    panel = R.Panel(fig, gs[0], sim, mission, f"attempt {n} · harness v{v}", "", ok)
    right_panel(fig.add_subplot(gs[1]), n, v, h, prev, t)
    fig.suptitle(f"{BODY} · {MID} — same mission, same model, only the harness changes", fontsize=15, fontweight="bold", color=INK)
    out = f"out/attempts/p{TAG}{n}.mp4"
    ts = R._timeline([panel])
    anim = animation.FuncAnimation(fig, lambda i: panel.draw(ts[i]) or [], frames=len(ts), blit=False)
    anim.save(out, writer=animation.FFMpegWriter(fps=R.FPS, bitrate=3000), dpi=110)
    plt.close(fig)
    files.append(out)
    prev = h
    print(out, v)
open(f"out/attempts/plist{TAG}.txt", "w").write("".join(f"file '{f.split('/')[-1]}'\n" for f in files))
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"plist{TAG}.txt", "-vf",
                "fps=24,format=yuv420p", "-c:v", "libx264", "-crf", "20", f"../attempts-with-params{TAG}.mp4"],
               cwd="out/attempts", check=True)
print(f"out/attempts-with-params{TAG}.mp4")
