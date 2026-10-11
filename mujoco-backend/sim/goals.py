"""Goal tasks: what a mission asks for, and how the LOCKED verifier checks it. Fixed code.

A goal is a list of tasks, all of which must hold at the end of the run:

  {"type": "deliver",     "item": "radio", "to": [x, y(, z)]}          item rests at `to`
  {"type": "visit",       "pos": [x, y(, z)], "robot": id?}            robot passed through the cell
  {"type": "inspect",     "pos": [x, y(, z)], "robot": id?}            robot ran scan() while at the cell
  {"type": "patrol",      "points": [[..], ..], "laps": 1, "robot": id?} points visited in order, `laps` times
  {"type": "return_home", "robot": id?}                                robot ends where it started
  {"type": "survey",      "area": [x0, y0, x1, y1], "coverage": 0.8}   scan() footprints cover >= coverage of the area
  {"type": "avoid",       "area": [x0, y0, x1, y1], "robot": id?}      robot never enters the area (keep-out)

Legacy goals ({"deliver": ...} / {"deliveries": [...]}) are read as deliver tasks.
Everything is checked from the physics log (frames, events), never from planner claims.
"""
from __future__ import annotations

TASK_TYPES = ("deliver", "visit", "inspect", "patrol", "return_home", "survey", "avoid")


def _inside(p, area) -> bool:
    x0, y0, x1, y1 = area
    return x0 <= p[0] < x1 and y0 <= (p[1] if len(p) > 1 else 0) < y1


def survey_points(area: list[int], rng: int, world=None) -> list[list[int]]:
    """Lawnmower waypoints whose scan footprints (square, radius `rng`) tile the area,
    each snapped to the nearest open ground cell when a world is given."""
    pts = _lawnmower(area, rng)
    if world is None:
        return pts
    from sim.world import T
    g = world.grid if world.dims < 3 else world.grid[:, :, 0]
    free = [(x, y) for x in range(g.shape[0]) for y in range(g.shape[1]) if T(int(g[x, y])) in (T.FLOOR, T.CHARGER)]
    out = []
    for x, y in pts:
        c = min(free, key=lambda f: abs(f[0] - x) + abs(f[1] - y))
        p = [*c, 0] if world.dims == 3 else list(c)
        if p not in out:
            out.append(p)
    return out


def _lawnmower(area: list[int], rng: int) -> list[list[int]]:
    x0, y0, x1, y1 = area
    step = max(1, 2 * rng + 1)

    def axis(lo, hi):
        cs = list(range(lo + min(rng, (hi - lo - 1) // 2), hi, step))
        if cs[-1] + rng < hi - 1:  # last footprint must reach the far edge
            cs.append(hi - 1 - rng)
        return cs
    pts, flip = [], False
    for x in axis(x0, x1):
        ys = axis(y0, y1)
        pts += [[x, y] for y in (ys[::-1] if flip else ys)]
        flip = not flip
    return pts


def tasks(goal: dict) -> list[dict]:
    if goal.get("tasks"):
        return goal["tasks"]
    ds = goal.get("deliveries") or ([goal["deliver"]] if goal.get("deliver") else [])
    return [{"type": "deliver", **d} for d in ds]


def _same(world, a, b) -> bool:
    a, b = tuple(a), tuple(b)
    if world.dims == 3 and (len(a) == 2 or len(b) == 2):
        return a[:2] == b[:2]
    return world.ground(a) == world.ground(b) if world.dims == 3 and (a[2] == 0 or b[2] == 0) else a == b


def _track(sim, rid: str | None) -> list[tuple]:
    """Ordered distinct positions of one robot (or any robot when rid is None)."""
    out = []
    for f in sim.frames:
        for r, st in f["robots"].items():
            if rid in (None, r):
                p = tuple(st["pos"])
                if not out or out[-1] != (r, p):
                    out.append((r, p))
    return out


def check(sim, task: dict) -> bool:
    w, kind, rid = sim.world, task.get("type"), task.get("robot")
    if kind == "deliver":
        item = sim.items.get(task["item"])
        return item is not None and item.pos is not None and w.ground(item.pos) == w.ground(tuple(task["to"])) \
            if w.dims == 3 else item is not None and item.pos is not None and tuple(item.pos) == tuple(task["to"])
    if kind == "visit":
        return any(_same(w, p, task["pos"]) for _, p in _track(sim, rid))
    if kind == "inspect":
        return any(e["kind"] == "scan" and rid in (None, e["robot"]) and _same(w, e["pos"], task["pos"])
                   for e in sim.events)
    if kind == "patrol":
        seq = [p for _, p in _track(sim, rid)]
        want = [tuple(p) for p in task["points"]] * int(task.get("laps", 1))
        i = 0
        for p in seq:
            if i < len(want) and _same(w, p, want[i]):
                i += 1
        return i == len(want)
    if kind == "return_home":
        robots = [rid] if rid else list(sim.robots)
        first, last = sim.frames[0]["robots"], sim.frames[-1]["robots"]
        return all(_same(w, last[r]["pos"], first[r]["pos"]) for r in robots)
    if kind == "survey":
        x0, y0, x1, y1 = task["area"]
        cells = {(x, y) for x in range(x0, x1) for y in range(y0, y1)}
        seen = set()
        for e in sim.events:
            if e["kind"] == "scan" and rid in (None, e["robot"]):
                k = sim.robots[e["robot"]].body.sensor_range
                px, py = e["pos"][0], e["pos"][1] if len(e["pos"]) > 1 else 0
                seen |= {c for c in cells if abs(c[0] - px) <= k and abs(c[1] - py) <= k}
        return len(seen) >= float(task.get("coverage", 1.0)) * len(cells)
    if kind == "avoid":
        return not any(_inside(p, task["area"]) for _, p in _track(sim, rid))
    raise ValueError(f"unknown task type {kind!r}; known: {TASK_TYPES}")


def describe(goal: dict) -> list[str]:
    """Plain-language task list for the planner."""
    out = []
    for t in tasks(goal):
        k = t["type"]
        out.append({"deliver": "carry $ITEM from $ITEM_POS to $DEST",
                    "visit": "pass through every point in $TARGETS",
                    "inspect": "at every point in $TARGETS, stop and scan()",
                    "patrol": f"visit $TARGETS in order, {t.get('laps', 1)} lap(s)",
                    "return_home": "end the mission back at $HOME",
                    "survey": "scan() at every point in $TARGETS (a lawnmower sweep of the area)",
                    "avoid": f"never enter the keep-out area {t.get('area')}"}[k])
    return list(dict.fromkeys(out))
