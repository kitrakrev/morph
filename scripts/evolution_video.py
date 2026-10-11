"""Render an 'evolution timeline' video from Atlas: inputs, then each version's changed parameters
(old -> new), prediction vs measured, scores and gate verdict. Usage: PYTHONPATH=. uv run python scripts/evolution_video.py"""
import json
import subprocess
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.store import db
from sim.missions import BODIES

BG, INK, MUTED, OK, BAD, ACC = "#0F1E19", "#F4F2EC", "#9FB5AC", "#4FC08D", "#E8735A", "#E8A33A"
OUT = "out/evo"


def scores(body, v):
    agg = {}
    for t in db().trials.find({"body": body, "harness_version": v}, {"split": 1, "mission_id": 1, "score": 1, "ts": 1}).sort("ts", 1):
        if "long_range" in t["mission_id"]:
            continue
        a = agg.setdefault(t["split"], {})
        a[t["mission_id"]] = t["score"]["success"]
    return {s: f"{sum(m.values())}/{len(m)}" for s, m in agg.items()}


def short(v, n=70):
    s = json.dumps(v, separators=(",", ":")) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n - 1] + "…"


def slide(path, title, sub, blocks, foot=""):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=BG)
    fig.text(0.05, 0.9, title, color=INK, fontsize=34, fontweight="bold", family="DejaVu Sans")
    fig.text(0.05, 0.84, sub, color=ACC, fontsize=17, family="DejaVu Sans")
    y = 0.75
    for label, text, color in blocks:
        fig.text(0.05, y, label, color=MUTED, fontsize=14, fontweight="bold", family="DejaVu Sans")
        for line in textwrap.wrap(text, 86) or [""]:
            fig.text(0.24, y, line.replace("$", r"\$"), color=color, fontsize=14, family="DejaVu Sans Mono")
            y -= 0.045
        y -= 0.02
    if foot:
        fig.text(0.05, 0.05, foot, color=MUTED, fontsize=13, family="DejaVu Sans")
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


def inputs_slide(path, body, lineage, goal):
    s = BODIES[body]
    slide(path, f"Start inputs — {lineage}", "what the user supplies (nothing else)", [
        ("robot", f"{body}: locomotion {short(s['locomotion'])}, mass {s['mass_kg']} kg, battery {s['battery_wh']} Wh, "
                  f"payload {s['payload_bay_kg']} kg, sensors {', '.join(s['sensors'])}", INK),
        ("terrain", "3D voxel city, medium: buildings 3–7 tall, no-fly zone below altitude 6, river, chargers, wind", INK),
        ("goal", goal, INK),
        ("planner model", "Claude Haiku via Claude Code headless — fixed for every version", INK),
        ("evolution model", "Claude Sonnet via Claude Code headless — proposes ONE bounded edit + a prediction", INK),
        ("locked", "simulator, true physics, verifier, held-out missions, no-fly + collision constraints", BAD),
    ], "Harness v1 is bootstrapped from the robot's structure alone.")


def version_slide(path, lineage, h):
    v, ev = h["version"], db().evaluations.find_one({"body": lineage, "version": h["version"]}) or {}
    sc = scores(lineage, v)
    verdict = h.get("verdict")
    blocks = []
    if v == 1:
        blocks.append(("harness v1", f"workflow {' → '.join(h['workflow']['nodes'])}", INK))
        blocks.append(("rules", " | ".join(h.get("rules", [])), INK))
        blocks.append(("tool_specs", short(h.get("tool_specs"), 100), INK))
        blocks.append(("beliefs", short(h.get("beliefs"), 100), INK))
    else:
        for d in (h.get("diff") if isinstance(h.get("diff"), list) else []):
            blocks.append((f"− {d['field']}", short(d["old"], 200), MUTED))
            blocks.append((f"+ {d['field']}", short(d["new"], 200), OK if verdict == "accepted" else BAD))
        if h.get("rationale"):
            blocks.append(("agent's reason", h["rationale"][:260], INK))
        if ev.get("prediction"):
            blocks.append(("predicted / measured", f"Δtrain {ev['prediction'].get('train_success_delta')} / "
                                                   f"{ev.get('measured', {}).get('train_success_delta')}", INK))
    blocks.append(("scores", f"train {sc.get('train', '—')} · regression {sc.get('regression', '—')} · "
                             f"held-out {sc.get('heldout', '—')}", INK))
    if v != 1:
        blocks.append(("gate", ("ACCEPTED — " if verdict == "accepted" else "REJECTED — ") + str(ev.get("reason", "")),
                       OK if verdict == "accepted" else BAD))
    slide(path, f"{lineage}  ·  harness v{v}", "bootstrap" if v == 1 else f"edit proposed from v{h.get('parent')}", blocks)


frames = []
plan = [("drone", "drone", [1, 8, 14, 17], "Deliver a 0.8 kg radio / 0.5 kg blanket to a target; zero safety violations, ≥20% battery reserve, before the deadline"),
        ("fixed_wing_vtol", "fixed_wing_vtol", [1, 2], "Same delivery goal; take-off and landing only on open cells; cruise at altitude ≥3")]
i = 0
for body, lineage, versions, goal in plan:
    i += 1
    p = f"{OUT}/s{i:02d}.png"
    inputs_slide(p, body, lineage, goal)
    frames.append((p, 6))
    for v in versions:
        h = db().harnesses.find_one({"body_type": lineage, "version": v})
        i += 1
        p = f"{OUT}/s{i:02d}.png"
        version_slide(p, lineage, h)
        frames.append((p, 6))
with open(f"{OUT}/list.txt", "w") as f:
    for p, d in frames:
        f.write(f"file '{p.split('/')[-1]}'\nduration {d}\n")
    f.write(f"file '{frames[-1][0].split('/')[-1]}'\n")
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
                "-vf", "fps=30,format=yuv420p,scale=1920:1080", "-c:v", "libx264", "-crf", "20", "../evolution-timeline.mp4"],
               cwd=OUT, check=True)
print("out/evolution-timeline.mp4", len(frames), "slides")
