"""MuJoCo rigid-body backend for rotorcraft. Fixed code; agents never import it.

Same skill interface as `sim.skills.Sim`: the planner still emits grid programs
(`fly_to`, `pick`, ...). Only the *execution* of each rotorcraft cell move changes:
instead of the closed-form grid energy model, a quadrotor (Skydio-X2-style
4-rotor layout, mass from the body structure + payload) is flown through the
cell in MuJoCo by a geometric position/attitude controller driving 4 thrust
motors. Buildings are real box geoms, so tracking error can hit them.

What comes out of real dynamics:
  * energy  - rotor power from momentum theory, P = sum k * u_i^1.5, integrated
              over the trajectory (payload, climbs, turns, gusts all cost thrust);
              calibrated so an unloaded hover-cell matches the grid model;
  * collisions - drone hull vs building contacts emit `collision` events;
  * wind    - seeded gusts whose strength grows with altitude.

Ground bodies and fixed-wing keep the grid model. Enable with
`mission["physics"] = "mujoco"` or env `MORPH_PHYSICS=mujoco`.
"""
from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

from sim import physics
from sim.skills import MAX_STEPS, Robot, Sim
from sim.world import T, Pos

G = 9.81
DT = 0.002
CRUISE = 1.0            # m/s through cells (1 cell = 1 m)
CELL_S = 1 / CRUISE     # seconds per cell
HOLD_S = 1.0            # max extra time to settle after landing
CATCH_M = 0.15          # "caught up" radius
GROUND_Z = 0.12         # landed body-centre height
RHO, FOM = 1.225, 0.6   # air density, rotor figure of merit
YAW_K = 0.0201          # rotor drag-torque / thrust ratio
MAX_THRUST_RATIO = 2.5  # per-rotor max thrust, as multiple of hover share
MAX_TILT_ACC = 0.7 * G  # horizontal accel cap (~35 deg tilt)
GUST_N_PER_Z = 0.015    # gust std-dev per altitude level, as fraction of weight
REC_EVERY = 20          # log a pose every 20 physics steps (25 fps of sim time)


def _grid_pos(sim: Sim, p: Pos) -> np.ndarray:
    """Cell centre in metres for any world dimension (ground bodies sit at z=0)."""
    x = p[0] + 0.5
    y = p[1] + 0.5 if len(p) > 1 else 0.5
    z = cell_center(p)[2] if len(p) == 3 and p[2] > 0 else 0.0
    return np.array([x, y, z])


def cell_center(p: Pos) -> np.ndarray:
    z = GROUND_Z if len(p) < 3 or p[2] == 0 else p[2] + 0.5
    return np.array([p[0] + 0.5, p[1] + 0.5, z])


def _hermite(p0, m0, p1, m1, s):
    """Cubic Hermite position, d/ds, d2/ds2 at s in [0, 1] (tangents m scaled by segment time)."""
    pos = (2*s**3 - 3*s**2 + 1) * p0 + (s**3 - 2*s**2 + s) * m0 + (-2*s**3 + 3*s**2) * p1 + (s**3 - s**2) * m1
    vel = (6*s**2 - 6*s) * p0 + (3*s**2 - 4*s + 1) * m0 + (-6*s**2 + 6*s) * p1 + (3*s**2 - 2*s) * m1
    acc = (12*s - 6) * p0 + (6*s - 4) * m0 + (-12*s + 6) * p1 + (6*s - 2) * m1
    return pos, vel, acc


def _arm(mass_kg: float) -> float:
    return 0.18 * (mass_kg / 2.5) ** (1 / 3)  # Skydio X2 ~0.18 m arm at ~2.5 kg


def world_xml(sim: Sim, mass_kg: float) -> str:
    """MJCF: ground plane, one box per building column, one free-flying quadrotor."""
    g = sim.world.grid
    boxes = []
    if sim.world.dims == 3:
        heights = (g == T.WALL).sum(axis=2)
        for x, y in np.argwhere(heights > 0):
            h = float(heights[x, y])
            boxes.append(f'<geom type="box" size="0.5 0.5 {h / 2}" pos="{x + 0.5} {y + 0.5} {h / 2}" '
                         f'contype="2" conaffinity="1"/>')
    a = _arm(mass_kg)
    rotors = [(a, a, -1), (-a, a, 1), (-a, -a, -1), (a, -a, 1)]
    sites = "".join(f'<site name="r{i}" pos="{x} {y} 0.02"/>' for i, (x, y, _) in enumerate(rotors))
    umax = MAX_THRUST_RATIO * mass_kg * G / 4
    motors = "".join(f'<motor site="r{i}" gear="0 0 1 0 0 {s * YAW_K}" ctrlrange="0 {umax}" ctrllimited="true"/>'
                     for i, (_, _, s) in enumerate(rotors))
    return f"""
<mujoco model="morph-quad">
  <option timestep="{DT}" gravity="0 0 -{G}"/>
  <worldbody>
    <geom type="plane" size="200 200 .1" contype="2" conaffinity="1"/>
    {''.join(boxes)}
    <body name="quad" pos="0 0 {GROUND_Z}">
      <freejoint/>
      <geom name="hull" type="box" size="{a} {a} 0.04" mass="{mass_kg}" contype="1" conaffinity="2"/>
      {sites}
    </body>
  </worldbody>
  <actuator>{motors}</actuator>
</mujoco>"""


def _noop(_engine):
    pass


def _yaw_quat(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([roll, pitch, yaw]), "xyz")
    return q


class Quad:
    """One robot's MuJoCo world plus its controller and energy meter."""

    on_step = staticmethod(_noop)

    def pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        """World position, orientation quaternion, mean rotor thrust ratio (for rendering)."""
        return self.data.qpos[:3].copy(), self.data.qpos[3:7].copy(), float(self.data.ctrl.mean() / (self.base_mass * G / 4))

    def __init__(self, sim: Sim, r: Robot, seed: int):
        self.base_mass = r.body.mass_kg
        self.model = mujoco.MjModel.from_xml_string(world_xml(sim, self.base_mass))
        self.data = mujoco.MjData(self.model)
        self.bid = self.model.body("quad").id
        self.hull = self.model.geom("hull").id
        self.base_inertia = self.model.body_inertia[self.bid].copy()
        self.data.qpos[:3] = cell_center(r.pos)
        mujoco.mj_forward(self.model, self.data)
        self.rng = np.random.default_rng(seed)
        a = _arm(self.base_mass)
        # rows: total thrust, roll torque, pitch torque, yaw torque
        mix = np.array([[1, 1, 1, 1], [a, a, -a, -a], [-a, a, a, -a], [-YAW_K, YAW_K, -YAW_K, YAW_K]])
        self.mix_inv = np.linalg.inv(mix)
        self.k_power = 1 / (FOM * math.sqrt(2 * RHO * math.pi * (0.6 * a) ** 2))
        self.hover_cell_j = 4 * self.k_power * (self.base_mass * G / 4) ** 1.5 * CELL_S
        self.peak_error = 0.0
        self.v_ref = np.zeros(3)  # reference velocity carried between cells

    def set_payload(self, kg: float):
        m = self.base_mass + kg
        self.model.body_mass[self.bid] = m
        self.model.body_inertia[self.bid] = self.base_inertia * m / self.base_mass

    def _control(self, p_t: np.ndarray, v_t: np.ndarray, a_t: np.ndarray) -> np.ndarray:
        d, m = self.data, self.model.body_mass[self.bid]
        R = d.xmat[self.bid].reshape(3, 3)
        w = d.qvel[3:6]  # freejoint angular velocity is in the body frame
        a_des = 8.0 * (p_t - d.qpos[:3]) + 5.0 * (v_t - d.qvel[:3]) + a_t
        h = np.linalg.norm(a_des[:2])
        if h > MAX_TILT_ACC:
            a_des[:2] *= MAX_TILT_ACC / h
        a_des[2] = max(a_des[2] + G, 0.2 * G)
        zb = a_des / np.linalg.norm(a_des)
        yb = np.cross(zb, [1.0, 0, 0])
        yb /= np.linalg.norm(yb)
        Rd = np.column_stack([np.cross(yb, zb), yb, zb])
        E = Rd.T @ R - R.T @ Rd
        e_r = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
        torque = self.model.body_inertia[self.bid] * (-300.0 * e_r - 35.0 * w)
        wrench = np.array([m * a_des @ R[:, 2], *torque])
        u = self.mix_inv @ wrench
        umax = self.model.actuator_ctrlrange[:, 1]
        if (u < 0).any() or (u > umax).any():  # saturated: give up yaw authority first
            wrench[3] = 0
            u = self.mix_inv @ wrench
        return np.clip(u, 0, umax)

    def fly_cell(self, a: Pos, b: Pos, after: Pos | None) -> tuple[float, bool]:
        """Fly a->b on a cubic Hermite segment. Returns (joules, hit_structure).

        Velocity is continuous across cells: the drone cruises straight through,
        stops to pivot at turns (`after` is the following waypoint), and
        stops on the ground layer or at the end of a route.
        """
        pa, pb = cell_center(a), cell_center(b)
        b_ground = len(b) < 3 or b[2] == 0
        landing = b_ground or after is None
        d0 = pb - pa
        if landing:
            v1 = np.zeros(3)
        else:
            # cruise straight through; stop and pivot on turns so the body never cuts a building corner
            v1 = d0 * CRUISE if np.allclose(d0, cell_center(after) - pb) else np.zeros(3)
        v0 = self.v_ref
        n = int(CELL_S / DT)
        alt = b[2] if len(b) == 3 else 0
        gust = self.rng.normal(0, GUST_N_PER_Z * alt * self.model.body_mass[self.bid] * G, size=3)
        gust[2] = 0
        joules, hit = 0.0, False
        for i in range(n + (int(HOLD_S / DT) if landing else 0)):
            s = min(1.0, (i + 1) / n)
            if s == 1.0 and np.linalg.norm(self.data.qpos[:3] - pb) < CATCH_M:
                break
            if landing and b_ground and self.data.qpos[2] < GROUND_Z + 0.05 and s > 0.5:
                break  # touchdown: disarm
            if i < n:
                p_t, v_t, a_t = _hermite(pa, v0 * CELL_S, pb, v1 * CELL_S, s)
                u = self._control(p_t, v_t / CELL_S, a_t / CELL_S**2)
            else:  # settle on the pad: no feedforward
                u = self._control(pb, np.zeros(3), np.zeros(3))
            self.data.ctrl[:] = u
            self.data.xfrc_applied[self.bid, :3] = gust
            mujoco.mj_step(self.model, self.data)
            self.on_step(self)
            joules += self.k_power * float(np.sum(u ** 1.5)) * DT
            for c in self.data.contact[: self.data.ncon]:
                if self.hull in (c.geom1, c.geom2) and min(c.geom1, c.geom2) > 0:  # geom 0 is the ground
                    hit = True
        if self.data.xmat[self.bid][8] < 0.5:  # tilted past 60 deg: crashed
            hit = True
        if b_ground:  # landing gear settles the airframe level on the pad
            self.data.qpos[2], self.data.qpos[3:7], self.data.qvel[:] = GROUND_Z, [1, 0, 0, 0], 0
            mujoco.mj_forward(self.model, self.data)
        err = float(np.linalg.norm(self.data.qpos[:3] - pb))
        self.peak_error = max(self.peak_error, err)
        self.v_ref = v1
        return joules, hit


# ---- ground and fixed-wing bodies --------------------------------------------
# Rolling / gait resistance coefficient per mode, and its multiplier per terrain.
CRR = {"wheel": 0.03, "tracks": 0.06, "legs": 0.10, "rail": 0.005}
TERRAIN_DRAG = {T.FLOOR: 1.0, T.CHARGER: 1.0, T.RAIL: 1.0, T.RUBBLE: 2.2, T.STAIRS: 2.0, T.NOFLY: 1.0}
DRIVE_ETA = 0.7              # motor + drivetrain efficiency
FW_CRUISE = 3.0              # m/s, fixed-wing cruise (1 cell per grid step)
FW_STALL = 2.0               # below this the wing lifts only (v/stall)^2 of the weight
FW_CD0, FW_K = 0.04, 0.05    # parasitic drag coeff (N per (m/s)^2 per kg), induced drag factor


class Mover:
    """Driven rigid body on slide joints (x, y[, z]) for ground and fixed-wing modes.

    Actuators are body-frame-free forces (an abstraction of wheels / tracks / legs /
    propeller+surfaces); dynamics, inertia, resistance and aero are integrated by MuJoCo.
    Energy = hotel power * time + positive actuator work / efficiency (+ VTOL lift power).
    """

    def __init__(self, sim: Sim, r: Robot, mode: str):
        self.mode, self.sim = mode, sim
        self.base_mass = r.body.mass_kg
        self.axes = 3 if mode == "fixed_wing" else min(sim.world.dims, 2)
        names = "xyz"[: self.axes]
        joints = "".join(f'<joint name="{a}" type="slide" axis="{" ".join("1" if a == b else "0" for b in "xyz")}"/>'
                         for a in names)
        fmax = 20 * self.base_mass * G
        acts = "".join(f'<motor joint="{a}" ctrlrange="{-fmax} {fmax}" ctrllimited="true"/>' for a in names)
        grav = f"0 0 -{G}" if self.axes == 3 else "0 0 0"
        self.model = mujoco.MjModel.from_xml_string(f"""
<mujoco><option timestep="{DT}" gravity="{grav}"/>
  <worldbody><body name="b">{joints}<geom type="sphere" size="0.2" mass="{self.base_mass}" contype="0" conaffinity="0"/></body></worldbody>
  <actuator>{acts}</actuator></mujoco>""")
        self.data = mujoco.MjData(self.model)
        self.bid = self.model.body("b").id
        self.data.qpos[:] = self._center(r.pos)
        self.v_ref = np.zeros(self.axes)
        self.cell_s = 1 / FW_CRUISE if mode == "fixed_wing" else 0.5
        self.k_hover = 1 / (FOM * math.sqrt(2 * RHO * 4 * math.pi * (0.6 * _arm(self.base_mass)) ** 2))
        # hotel power (compute, sensors, idle drive) sized so an unloaded steady floor cell
        # costs the grid model's BASE_WH; everything mechanical on top is MuJoCo work
        self.hotel_w = 0.0
        floor_cells = physics.STEP_COST.get(mode, {}).get(T.FLOOR, 1)
        self.hotel_w = max(physics.BASE_WH[mode] * 3600 - self._calibrate(floor_cells), 0.0) / (self.cell_s * floor_cells)

    on_step = staticmethod(_noop)

    def set_payload(self, kg: float):
        self.model.body_mass[self.bid] = self.base_mass + kg

    def pose(self) -> tuple[np.ndarray, np.ndarray, float]:
        """World position, heading quaternion (+ pitch/bank for fixed-wing), speed (for rendering)."""
        p, v = np.zeros(3), np.zeros(3)
        p[: self.axes], v[: self.axes] = self.data.qpos, self.data.qvel
        if self.axes < 3:
            p[2] = 0.0
        speed = float(np.linalg.norm(v[:2]))
        if speed > 0.05:
            self.heading = math.atan2(v[1], v[0])
        yaw = getattr(self, "heading", 0.0)
        pitch = roll = 0.0
        if self.mode == "fixed_wing" and speed > 0.5:
            pitch = -math.atan2(v[2], speed)
            prev = getattr(self, "_prev_yaw", yaw)
            roll = float(np.clip(-(yaw - prev + math.pi) % (2 * math.pi) + math.pi, -0.02, 0.02)) * 30
            self._prev_yaw = yaw
        return p, _yaw_quat(yaw, pitch, roll), speed

    def _center(self, p: Pos) -> np.ndarray:
        c = np.array([p[0] + 0.5] + ([p[1] + 0.5] if len(p) > 1 else []) + ([p[2] + 0.5] if len(p) > 2 else []))
        return c[: self.axes]

    def _resist(self, v: np.ndarray, terrain: T) -> np.ndarray:
        m = self.model.body_mass[self.bid]
        if self.mode == "fixed_wing":
            speed = float(np.linalg.norm(v[:2]))
            lift = m * G * min(1.0, (speed / FW_STALL) ** 2)
            drag = FW_CD0 * self.base_mass * speed ** 2 + FW_K * (m * G) ** 2 / max(speed, FW_STALL) ** 2 / self.base_mass
            f = np.array([0.0, 0.0, lift])
            if speed > 1e-6:
                f[:2] -= drag * v[:2] / speed
            return f
        rr = CRR[self.mode] * TERRAIN_DRAG.get(terrain, 1.0) * m * G
        return -rr * v / (np.linalg.norm(v) + 0.05)  # smoothed Coulomb rolling resistance

    def run(self, a: Pos, b: Pos, after: Pos | None, terrain: T, cells: int) -> float:
        """Drive a->b over `cells` grid steps; returns joules drawn."""
        pa, pb = self._center(a), self._center(b)
        d0 = pb - pa
        straight = after is not None and np.allclose(d0, self._center(after) - pb)
        if self.mode == "fixed_wing" and after is not None and d0[2] == 0:
            d1 = self._center(after) - pb
            v1 = FW_CRUISE * (d0 + d1) / (np.linalg.norm(d0 + d1) + 1e-9) if d1[2] == 0 else np.zeros(3)
        else:
            v1 = d0 / (self.cell_s * cells) if straight else np.zeros(self.axes)
        T_s = self.cell_s * cells
        n = int(T_s / DT)
        joules = self.hotel_w * T_s
        for i in range(n):
            s = (i + 1) / n
            p_t, v_t, a_t = _hermite(pa, self.v_ref * T_s, pb, v1 * T_s, s)
            v = self.data.qvel.copy()
            ext = self._resist(v, terrain)
            m = self.model.body_mass[self.bid]
            f = m * (40.0 * (p_t - self.data.qpos) + 12.0 * (v_t / T_s - v) + a_t / T_s**2) - ext
            if self.axes == 3:
                f[2] += m * G
            self.data.qfrc_applied[:] = ext
            self.data.ctrl[:] = f
            mujoco.mj_step(self.model, self.data)
            self.on_step(self)
            joules += max(float(f @ v), 0.0) / DRIVE_ETA * DT
            if self.mode == "fixed_wing":  # VTOL lift rotors carry whatever the wing does not
                unsupported = max(m * G - ext[2], 0.0)
                joules += self.k_hover * unsupported ** 1.5 * DT
        self.v_ref = v1
        if self.axes == 3 and len(b) == 3 and b[2] == 0:
            self.v_ref = np.zeros(3)
        return joules

    def _calibrate(self, cells: int) -> float:
        """Mechanical joules for one unloaded floor cell at steady speed."""
        q0, v0 = self.data.qpos.copy(), self.v_ref.copy()
        z = (0,) * (self.axes - 1)
        if self.mode == "fixed_wing":
            z = (0, 4)
        a, b, c = (0, *z), (1, *z), (2, *z)
        self.data.qpos[:] = self._center(a)
        self.data.qvel[:] = (self._center(b) - self._center(a)) / (self.cell_s * cells)
        self.v_ref = self.data.qvel.copy()
        j = self.run(a, b, c, T.FLOOR, cells)
        self.data.qpos[:], self.data.qvel[:], self.v_ref = q0, 0, v0
        mujoco.mj_forward(self.model, self.data)
        return j


class MujocoSim(Sim):
    """Sim whose rotorcraft moves are executed by MuJoCo rigid-body dynamics."""

    @property
    def _quads(self) -> dict[str, Quad]:
        return self.__dict__.setdefault("quads", {})

    def clone(self) -> "MujocoSim":
        quads = self.__dict__.pop("quads", {})
        try:
            return super().clone()  # MuJoCo state is re-created lazily from robot.pos
        finally:
            self.__dict__["quads"] = quads

    # ---- recording (for sim.render3d) -------------------------------------------
    @property
    def trace(self) -> list[dict]:
        """Real-time pose log: one entry per REC_DT seconds of simulated MuJoCo time."""
        return self.__dict__.setdefault("trace_", [])

    def _clock(self) -> float:
        return self.__dict__.get("clock", 0.0)

    def _hook(self, r: Robot, mode: str):
        def on_step(eng):
            self.__dict__["clock"] = self._clock() + DT
            self.__dict__["tick"] = self.__dict__.get("tick", 0) + 1
            if self.__dict__["tick"] % REC_EVERY == 0:
                self._record(r, mode, *eng.pose())
        return on_step

    def _record(self, r: Robot, mode: str, p, q, act: float, label: str = ""):
        self.trace.append({"t": round(self._clock(), 3), "robot": r.id, "kind": r.body.kind, "mode": mode,
                           "pos": [round(float(x), 3) for x in p], "quat": [round(float(x), 4) for x in q],
                           "act": round(act, 3), "battery": round(r.battery / r.body.battery_wh, 3),
                           "carrying": list(r.carrying), "label": label})

    def _hold(self, rid: str, seconds: float, label: str):
        """Log a pause (pick / drop / charge) at the robot's last recorded pose."""
        r = self.robots[rid]
        last = next((f for f in reversed(self.trace) if f["robot"] == rid), None)
        p = last["pos"] if last else list(_grid_pos(self, r.pos))
        q = last["quat"] if last else [1, 0, 0, 0]
        mode = last["mode"] if last else r.body.modes[0]
        for _ in range(int(seconds / (REC_EVERY * DT))):
            self.__dict__["clock"] = self._clock() + REC_EVERY * DT
            self._record(r, mode, p, q, 0.0, label)

    def run_call(self, call: dict) -> dict:
        res = super().run_call(call)
        skill, rid = call.get("skill"), call.get("robot")
        if rid in self.robots and skill in ("pick", "drop", "charge", "handoff", "scan"):
            label = f"{skill} {call.get('args', {}).get('item', '')}".strip() + ("" if res.get("ok") else " (failed)")
            self._hold(rid, 1.6 if skill == "charge" else 0.8, label)
        return res

    def mover(self, r: Robot, mode: str) -> Mover:
        key = f"{r.id}:{mode}"
        m = self._quads.get(key)
        if m is None:
            m = self._quads[key] = Mover(self, r, mode)
            m.on_step = self._hook(r, mode)
        return m

    def quad(self, r: Robot) -> Quad:
        q = self._quads.get(r.id)
        if q is None:
            q = self._quads[r.id] = Quad(self, r, self.world.seed * 1000 + len(self._quads))
            q.on_step = self._hook(r, "flight")
        return q

    def _with_route(self, skill: str, rid: str, *args) -> dict:
        # dry-run the same skill on grid physics to learn the waypoint route (lookahead for corners)
        dry = Sim(self.world, copy.deepcopy(self.robots), copy.deepcopy(self.items))
        getattr(dry, skill)(rid, *args)
        route = [tuple(f["robots"][rid]["pos"]) for f in dry.frames]
        self.__dict__["route"] = {p: route[i + 1] if i + 1 < len(route) else None for i, p in enumerate(route)}
        try:
            return getattr(super(), skill)(rid, *args)
        finally:
            self.__dict__.pop("route", None)

    def fly_to(self, rid: str, pos: list[int], alt: int = 0, avoid_nofly: bool = False) -> dict:
        return self._with_route("fly_to", rid, pos, alt, avoid_nofly)

    def move_to(self, rid: str, pos: list[int]) -> dict:
        return self._with_route("move_to", rid, pos)

    def _advance(self, r: Robot, nxt: Pos, mode: str) -> bool:
        if not r.alive:
            return False
        if mode != "flight":
            return self._advance_mover(r, nxt, mode)
        q = self.quad(r)
        q.set_payload(self.payload(r))
        if q.model.actuator_ctrlrange[:, 1].sum() < (q.base_mass + self.payload(r)) * G:
            self._fail(r.id, "fly_to", f"rotors cannot lift {q.base_mass + self.payload(r):.1f} kg")
            return False
        joules, hit = q.fly_cell(r.pos, nxt, self.__dict__.get("route", {}).get(nxt))
        # calibrate: unloaded hover-cell == grid model (BASE + HOVER Wh), so batteries keep meaning
        wh = joules / q.hover_cell_j * (physics.BASE_WH["flight"] + physics.HOVER_WH)
        return self._commit(r, nxt, mode, wh, hit)

    def _advance_mover(self, r: Robot, nxt: Pos, mode: str) -> bool:
        mv = self.mover(r, mode)
        mv.set_payload(self.payload(r))
        cells = physics.step_cost(self.world, mode, nxt)
        wh = mv.run(r.pos, nxt, self.__dict__.get("route", {}).get(nxt), self.world.at(nxt), cells) / 3600
        return self._commit(r, nxt, mode, wh, False)

    def _commit(self, r: Robot, nxt: Pos, mode: str, wh: float, hit: bool) -> bool:
        r.battery -= wh
        r.drained_wh += wh
        self.t += physics.step_cost(self.world, mode, nxt)
        r.pos = nxt
        if self.world.at(nxt) == T.NOFLY:
            self.emit("nofly", r.id, pos=list(nxt))
        if hit:
            self.emit("collision", r.id, other="structure", pos=list(nxt))
        for o in self.robots.values():
            if o.id != r.id and o.alive and o.pos == r.pos:
                self.emit("collision", r.id, other=o.id, pos=list(nxt))
        if r.battery <= 0:
            r.battery, r.alive = 0.0, False
            self.emit("battery_death", r.id, pos=list(nxt), airborne=r.airborne)
        self.snapshot()
        return r.alive and self.t < MAX_STEPS
