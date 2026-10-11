"""Skill primitives and the simulation state they act on. Fixed code.

The planner emits a program: a list of {"robot": id, "skill": name, "args": {...}}.
Only the primitives in PRIMITIVES exist; composed skills are sequences of these.
"""
from __future__ import annotations

import copy
import heapq
from dataclasses import dataclass, field

from sim import physics
from sim.physics import Body
from sim.world import T, World, Pos

PRIMITIVES = ("move_to", "fly_to", "pick", "drop", "charge", "scan", "handoff")
MAX_STEPS = 2000


@dataclass
class Robot:
    id: str
    body: Body
    pos: Pos
    battery: float
    carrying: list[str] = field(default_factory=list)
    alive: bool = True
    drained_wh: float = 0.0

    @property
    def airborne(self) -> bool:
        return len(self.pos) == 3 and self.pos[2] > 0


@dataclass
class Item:
    id: str
    weight_kg: float
    pos: Pos | None  # None while carried


@dataclass
class Sim:
    world: World
    robots: dict[str, Robot]
    items: dict[str, Item]
    t: int = 0
    events: list[dict] = field(default_factory=list)
    frames: list[dict] = field(default_factory=list)

    def clone(self) -> "Sim":
        return copy.deepcopy(self)

    def emit(self, kind: str, robot: str, **kw):
        self.events.append({"t": self.t, "kind": kind, "robot": robot, **kw})

    def snapshot(self):
        self.frames.append({"t": self.t, "robots": {r.id: {"pos": list(r.pos), "battery": round(r.battery, 2),
                                                            "alive": r.alive, "carrying": list(r.carrying)}
                                                     for r in self.robots.values()}})

    def payload(self, r: Robot) -> float:
        return sum(self.items[i].weight_kg for i in r.carrying)

    # ---- low-level motion -------------------------------------------------
    def _advance(self, r: Robot, nxt: Pos, mode: str) -> bool:
        if not r.alive:
            return False
        steps = physics.step_cost(self.world, mode, nxt)
        wh = physics.move_energy(r.body, mode, r.pos, nxt, self.payload(r))
        if mode == "flight":
            wh += physics.HOVER_WH * steps * (1 + physics.WIND_PER_Z * (nxt[2] if len(nxt) == 3 else 0))
        r.battery -= wh
        r.drained_wh += wh
        self.t += steps
        r.pos = nxt
        if self.world.at(nxt) == T.NOFLY:
            self.emit("nofly", r.id, pos=list(nxt))
        for o in self.robots.values():
            if o.id != r.id and o.alive and o.pos == r.pos:
                self.emit("collision", r.id, other=o.id, pos=list(nxt))
        if r.battery <= 0:
            r.battery, r.alive = 0.0, False
            self.emit("battery_death", r.id, pos=list(nxt), airborne=r.airborne)
        self.snapshot()
        return r.alive and self.t < MAX_STEPS

    def _astar(self, start: Pos, goal: Pos, mode: str, layer: int | None = None,
               avoid: tuple = ()) -> list[Pos] | None:
        """A* over cells passable for `mode`. In 3D flight, `layer` pins the altitude.
        `avoid` lists terrain types to route around (e.g. NOFLY)."""
        def ok(p):
            if layer is not None and p[2] != layer:
                return False
            if avoid and self.world.inside(p) and self.world.at(p) in avoid:
                return False
            return physics.passable(self.world, mode, p)

        def h(p):
            return sum(abs(a - b) for a, b in zip(p, goal))

        if not ok(goal):
            return None
        frontier = [(h(start), 0, start)]
        came, cost = {start: None}, {start: 0}
        while frontier:
            _, g, cur = heapq.heappop(frontier)
            if cur == goal:
                path = []
                while cur != start:
                    path.append(cur)
                    cur = came[cur]
                return path[::-1]
            for n in physics.neighbors(self.world, cur):
                if not ok(n):
                    continue
                ng = g + physics.step_cost(self.world, mode, n)
                if ng < cost.get(n, 1 << 30):
                    cost[n], came[n] = ng, cur
                    heapq.heappush(frontier, (ng + h(n), ng, n))
        return None

    # ---- primitives -------------------------------------------------------
    def move_to(self, rid: str, pos: list[int]) -> dict:
        r = self.robots[rid]
        mode = next((m for m in r.body.modes if m != "flight"), None)
        if mode is None:
            return self._fail(rid, "move_to", "body cannot move on ground")
        goal = tuple(pos)
        if self.world.dims == 3 and len(goal) == 2:
            goal = (*goal, 0)
        path = self._astar(r.pos, goal, mode)
        if path is None:
            return self._fail(rid, "move_to", f"no {mode} path to {list(goal)}")
        for p in path:
            if not self._advance(r, p, mode):
                return {"ok": False, "reason": "stopped", "pos": list(r.pos)}
        return {"ok": True, "pos": list(r.pos)}

    def fly_to(self, rid: str, pos: list[int], alt: int = 0, avoid_nofly: bool = False) -> dict:
        r = self.robots[rid]
        if "flight" not in r.body.modes:
            return self._fail(rid, "fly_to", "body cannot fly")
        avoid = (T.NOFLY,) if avoid_nofly else ()
        if self.world.dims < 3:  # 1D/2D: flight ignores ground terrain except walls
            path = self._astar(r.pos, tuple(pos), "flight", avoid=avoid)
            legs = [path] if path is not None else None
        else:
            x, y = pos[0], pos[1]
            z_end = pos[2] if len(pos) == 3 else 0
            alt = max(1, min(int(alt), self.world.size[2] - 1))
            up = [(*r.pos[:2], z) for z in range(r.pos[2] + 1, alt + 1)] if r.pos[2] < alt else \
                 [(*r.pos[:2], z) for z in range(r.pos[2] - 1, alt - 1, -1)]
            cruise = self._astar((*r.pos[:2], alt), (x, y, alt), "flight", layer=alt, avoid=avoid)
            down = [(x, y, z) for z in range(alt - 1, z_end - 1, -1)]
            legs = None if cruise is None else [up, cruise, down]
        if legs is None:
            return self._fail(rid, "fly_to", f"no flight path at alt {alt}")
        for leg in legs:
            for p in leg:
                if not physics.passable(self.world, "flight", p):
                    return self._fail(rid, "fly_to", f"blocked at {list(p)}")
                if not self._advance(r, p, "flight"):
                    return {"ok": False, "reason": "stopped", "pos": list(r.pos)}
        return {"ok": True, "pos": list(r.pos)}

    def pick(self, rid: str, item: str) -> dict:
        r, it = self.robots[rid], self.items.get(item)
        if it is None or it.pos is None or self.world.ground(it.pos) != self.world.ground(r.pos) or r.airborne:
            return self._fail(rid, "pick", f"{item} not here")
        r.carrying.append(item)
        it.pos = None
        if self.payload(r) > r.body.payload_max_kg:
            self.emit("payload_violation", rid, kg=self.payload(r), max=r.body.payload_max_kg)
        self.t += 1
        self.snapshot()
        return {"ok": True}

    def drop(self, rid: str, item: str) -> dict:
        r = self.robots[rid]
        if item not in r.carrying:
            return self._fail(rid, "drop", f"not carrying {item}")
        r.carrying.remove(item)
        self.items[item].pos = r.pos
        self.emit("drop", rid, item=item, pos=list(r.pos))
        self.t += 1
        self.snapshot()
        return {"ok": True}

    def charge(self, rid: str) -> dict:
        r = self.robots[rid]
        if self.world.at(self.world.ground(r.pos)) != T.CHARGER or r.airborne:
            return self._fail(rid, "charge", "not on a charger")
        while r.battery < r.body.battery_wh:
            r.battery = min(r.body.battery_wh, r.battery + physics.CHARGE_WH_PER_STEP)
            self.t += 1
        self.snapshot()
        return {"ok": True, "battery": r.battery}

    def scan(self, rid: str) -> dict:
        r = self.robots[rid]
        k = r.body.sensor_range
        sl = tuple(slice(max(0, c - k), c + k + 1) for c in r.pos)
        patch = self.world.grid[sl]
        self.t += 1
        return {"ok": True, "obstacles": int((patch == T.WALL).sum()), "nofly": int((patch == T.NOFLY).sum())}

    def handoff(self, rid: str, robot_id: str, item: str | None = None) -> dict:
        a, b = self.robots[rid], self.robots.get(robot_id)
        if b is None or not b.alive or sum(abs(x - y) for x, y in zip(a.pos, b.pos)) > 1:
            return self._fail(rid, "handoff", f"{robot_id} not adjacent")
        items = [item] if item else list(a.carrying)
        for i in items:
            a.carrying.remove(i)
            b.carrying.append(i)
        self.emit("handoff", rid, to=robot_id, items=items)
        self.t += 1
        self.snapshot()
        return {"ok": True}

    def _fail(self, rid: str, skill: str, reason: str) -> dict:
        self.emit("skill_error", rid, skill=skill, reason=reason)
        return {"ok": False, "reason": reason}

    def run_call(self, call: dict) -> dict:
        name = call.get("skill")
        if name not in PRIMITIVES:
            return self._fail(call.get("robot", "?"), str(name), "unknown skill")
        rid = call.get("robot")
        if rid not in self.robots:
            return self._fail(str(rid), name, "unknown robot")
        try:
            return getattr(self, name)(rid, **call.get("args", {}))
        except TypeError as e:
            return self._fail(rid, name, f"bad args: {e}")
