"""Evolution video: the same held-out mission replayed in MuJoCo under each accepted harness version,
with the inputs (robot, terrain, goal) and the harness parameters side by side; changed fields glow.

    PYTHONPATH=. uv run python scripts/render_evolution.py drone heldout-drone-903 [1 8 17]
"""
import subprocess
import sys
import textwrap

from PIL import Image, ImageDraw

from engine.store import db
from sim import tier2, verifier
from sim.goals import tasks
from sim.mission import build_sim
from sim.missions import make_mission
from sim.render3d import H, _font, render3d

PW = 640
BG, FG, DIM, HI, OK, BAD = (14, 18, 28), (226, 232, 240), (130, 142, 160), (255, 196, 64), (70, 210, 120), (240, 80, 70)


def harness_lines(h: dict) -> list[tuple[str, str]]:
    """(key, text) rows describing the evolvable harness fields."""
    rows = [("workflow", " > ".join(h["workflow"]["nodes"]))]
    b = h.get("beliefs", {})
    rows.append(("beliefs", f"Wh/cell {b.get('wh_per_cell')}  reserve {b.get('reserve_frac')}"))
    rows.append(("tool_specs", ", ".join(f"{k}{v}" for k, v in h.get("tool_specs", {}).items())))
    c = {k: v for k, v in h.get("constraints", {}).items()}
    rows.append(("constraints", ", ".join(f"{k}={v}" for k, v in c.items())))
    for i, r in enumerate(h.get("rules", [])):
        rows.append((f"rule{i}", r))
    if h.get("skills"):
        rows.append(("skills", ", ".join(s.get("name", "?") for s in h["skills"] if isinstance(s, dict))))
    return rows


def make_panel(mission, h, prev, score, recorded, version_idx, n_versions):
    f_h, f_b, f_s = _font(22), _font(15), _font(13)
    body = build_sim({**mission, "physics": "grid"}).robots[mission["robots"][0]["id"]].body
    st = mission["robots"][0]["structure"]
    t = mission["terrain"]
    goal = mission["goal"]
    now = dict(harness_lines(h))
    before = dict(harness_lines(prev)) if prev else {}

    def draw(frame, progress):
        im = Image.new("RGB", (PW, H), BG)
        d = ImageDraw.Draw(im)
        y = 16

        def head(txt, col=FG):
            nonlocal y
            d.text((20, y), txt, font=f_h, fill=col)
            y += 32

        def line(txt, col=FG, font=f_b, indent=20, width=66):
            nonlocal y
            for part in textwrap.wrap(txt, width) or [""]:
                d.text((indent, y), part, font=font, fill=col)
                y += 19
        head("INPUT  (user supplied)", DIM)
        line(f"robot: {body.kind}{' / ' + st['model'] if st.get('model') else ''}  locomotion {st.get('locomotion')}")
        line(f"       {st.get('mass_kg')} kg, battery {st.get('battery_wh')} Wh, payload {st.get('payload_bay_kg')} kg")
        line(f"terrain: {t.get('dims', '?')}D city, difficulty {t.get('difficulty', 'map')}, seed {t.get('seed', '-')}")
        y += 6
        head("GOAL", DIM)
        for task in tasks(goal):
            what = f"{task['type']} {task.get('item', '')} -> {task.get('to', task.get('pos', ''))}".strip()
            line("- " + what)
        line(f"- constraints {goal.get('constraints')}, reserve {goal.get('reserve_frac', 0)}, "
             f"deadline {goal.get('deadline_steps')} steps", DIM)
        y += 6
        head(f"HARNESS v{h['version']}  ({version_idx + 1}/{n_versions})", HI)
        for k, v in now.items():
            changed = prev is not None and before.get(k) != v
            line(("* " if changed else "  ") + f"{k}: {v}", HI if changed else FG, f_s, width=70)
        if h.get("rationale"):
            y += 4
            line("edit rationale: " + h["rationale"], DIM, f_s, width=70)
        y = H - 120
        d.line([20, y - 10, PW - 20, y - 10], fill=(50, 60, 80), width=1)
        line(f"recorded (grid physics): {'SUCCESS' if recorded['success'] else 'FAIL'}"
             f"  {recorded['energy_wh']} Wh", DIM)
        if progress >= 0.999:
            ok = score["success"]
            why = "" if ok else " - " + ", ".join([k for k, v in score["safety_violations"].items() if v] +
                                                  (["reserve breach"] if score["reserve_breach"] else []) +
                                                  ([] if score["delivered"] else ["not delivered"]))
            head(f"MuJoCo result: {'GOAL MET' if ok else 'FAILED'}{why}", OK if ok else BAD)
            line(f"{score['energy_wh']} Wh, {score['time']} steps, violations {score['violations_total']}", FG)
        else:
            head("MuJoCo: running ...", DIM)
        return im
    return draw


if __name__ == "__main__":
    body, mission_id, *vs = sys.argv[1:]
    versions = [int(v) for v in vs] or sorted(
        x["version"] for x in db().harnesses.find({"body_type": body, "verdict": {"$in": ["bootstrap", "accepted"]}}))
    parts = mission_id.split("-")
    mission = make_mission(body, int(parts[2]), parts[0], difficulty=parts[3] if len(parts) > 4 else "medium",
                           mtype=parts[4] if len(parts) > 4 else "single")
    clips, prev = [], None
    for i, v in enumerate(versions):
        h = db().harnesses.find_one({"body_type": body, "version": v}, {"_id": 0})
        trial = db().trials.find_one({"body": body, "mission_id": mission_id, "harness_version": v}, sort=[("ts", -1)])
        if not h or not trial:
            print("skip v", v, "(no harness or trial)")
            continue
        sim = tier2.simulate({**mission, "physics": "mujoco"}, trial["program"])
        score = verifier.score(mission, sim)
        out = f"out/evo-{mission_id}-v{v}.mp4"
        render3d(sim, out, f"{body} harness v{v} | {mission_id} (held-out)",
                 panel=make_panel(mission, h, prev, score, trial["score"], i, len(versions)), max_seconds=22)
        print(out, "MuJoCo", "GOAL MET" if score["success"] else "FAILED", score["energy_wh"], "Wh")
        clips.append(out)
        prev = h
    lst = "out/evo-list.txt"
    with open(lst, "w") as f:
        f.writelines(f"file '{c.split('/', 1)[1]}'\n" for c in clips)
    final = f"out/evolution-{mission_id}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", final],
                   check=True)
    print(final)
