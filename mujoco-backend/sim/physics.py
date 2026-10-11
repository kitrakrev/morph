"""TRUE physics for Morph. Fixed code. Agents never import this module.

The harness only sees outcomes (trials, verifier scores). Its beliefs about
speed/energy live in the harness doc and may be wrong; that gap is what the
evolution loop has to close.
"""
from __future__ import annotations

from dataclasses import dataclass

from sim.world import T, World, Pos

MODES = ("wheel", "tracks", "legs", "flight", "fixed_wing", "rail")

# steps needed to cross one cell, per mode and terrain (None = impassable)
STEP_COST: dict[str, dict[T, int | None]] = {
    "wheel": {T.FLOOR: 1, T.CHARGER: 1, T.RAIL: 1, T.RUBBLE: None, T.STAIRS: None, T.WATER: None},
    "tracks": {T.FLOOR: 1, T.CHARGER: 1, T.RAIL: 1, T.RUBBLE: 2, T.STAIRS: None, T.WATER: None},
    "legs": {T.FLOOR: 2, T.CHARGER: 2, T.RAIL: 2, T.RUBBLE: 3, T.STAIRS: 2, T.WATER: None},
    "rail": {T.RAIL: 1, T.CHARGER: 1},
    "flight": {T.AIR: 1, T.FLOOR: 1, T.CHARGER: 1, T.RAIL: 1, T.RUBBLE: 1, T.STAIRS: 1, T.WATER: 1},
}

# Wh drained per cell moved at zero payload
BASE_WH: dict[str, float] = {"wheel": 0.15, "tracks": 0.25, "legs": 0.35, "rail": 0.05, "flight": 0.5, "fixed_wing": 0.18}
HOVER_WH = 0.08          # flight: per step spent airborne regardless of motion
WIND_PER_Z = 0.12        # flight: hover/wind drain grows 12% per altitude level
CLIMB_WH = 0.6           # flight: per +1 altitude
CHARGE_WH_PER_STEP = 6.0
# fixed-wing plugin (human-written physics for a new body type)
FW_MIN_ALT = 3           # cannot cruise below this altitude
FW_TURN_WH = 0.6         # extra drain per 90-degree heading change in cruise (turn radius)
AIR_MODES = ("flight", "fixed_wing")


@dataclass(frozen=True)
class Body:
    """Derived from user-supplied structure. Truth, not belief."""
    kind: str
    modes: tuple[str, ...]
    mass_kg: float
    battery_wh: float
    payload_max_kg: float
    sensor_range: int
    model: str = ""  # real robot this body is modelled on (render only), e.g. "unitree_go2"

    @staticmethod
    def from_structure(s: dict) -> "Body":
        if s.get("model_file") and not s.get("locomotion"):
            from sim.robot_import import resolve  # robot supplied as a MuJoCo model file
            s = resolve(s)
        loco = s.get("locomotion", {})
        if "fixed_wing" in loco:
            kind, modes = "fixed_wing_vtol", ("fixed_wing",)
        elif "rotors" in loco and "wheels" in loco:  # drive-and-fly body (e.g. Caltech M4 Morphobot)
            kind, modes = "hybrid", ("wheel", "flight")
        elif "rotors" in loco:
            kind, modes = "drone", ("flight",)
        elif "legs" in loco:
            kind, modes = {2: "humanoid", 4: "quadruped"}.get(loco["legs"], "hexapod"), ("legs",)
        elif loco.get("rail"):
            kind, modes = "railcart", ("rail",)
        elif "tracks" in loco:
            kind, modes = "tracked", ("tracks",)
        elif "wheels" in loco:
            kind, modes = "rover", ("wheel",)
        else:
            raise ValueError(f"unknown locomotion {loco}")
        lidar = [x for x in s.get("sensors", []) if str(x).startswith("lidar_")]
        rng = int(lidar[0].split("_")[1].rstrip("m")) if lidar else 5
        return Body(kind, modes, float(s.get("mass_kg", 10)), float(s.get("battery_wh", 100)),
                    float(s.get("payload_bay_kg", 5)), rng, str(s.get("model", "")))


def payload_factor(body: Body, mode: str, payload_kg: float) -> float:
    """Hidden truth: flight energy grows quadratically with payload ratio."""
    r = payload_kg / max(body.mass_kg, 0.1)
    return 1 + 4 * r * r if mode == "flight" else 1 + r  # fixed-wing: mild, linear


def passable(world: World, mode: str, p: Pos) -> bool:
    if not world.inside(p):
        return False
    t = world.at(p)
    if mode in AIR_MODES:
        return t != T.WALL  # nofly is physically flyable; the verifier flags it
    if world.dims == 3 and p[2] != 0:
        return False
    if t == T.NOFLY:
        return mode == "flight"  # physically possible; verifier flags it
    return STEP_COST[mode].get(t) is not None


def step_cost(world: World, mode: str, p: Pos) -> int:
    t = world.at(p)
    if mode in AIR_MODES or t == T.NOFLY:
        return 1
    return STEP_COST[mode][t]


def move_energy(body: Body, mode: str, a: Pos, b: Pos, payload_kg: float) -> float:
    wh = BASE_WH[mode] * payload_factor(body, mode, payload_kg)
    if mode in AIR_MODES and len(b) == 3 and b[2] > a[2]:
        wh += CLIMB_WH * (b[2] - a[2]) * payload_factor(body, mode, payload_kg)
    return wh


def neighbors(world: World, p: Pos):
    for axis in range(world.dims):
        for d in (-1, 1):
            q = list(p)
            q[axis] += d
            yield tuple(q)
