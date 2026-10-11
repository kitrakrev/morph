"""3D MuJoCo video renderer. Fixed code.

Replays the real-time pose trace that `MujocoSim` records while it executes a
program (`sim.trace`) inside a visual-only MuJoCo scene: the city (terrain tiles,
extruded buildings, river, rubble, stairs, rail, chargers, no-fly volume) plus an
articulated model per body kind whose parts are driven from the trace -
rotors spin with thrust, wheels and tracks roll with speed, legs follow a
trot / tripod gait. Robots are mocap bodies; nothing here affects scoring.

    from sim.render3d import render3d
    render3d(sim, "out/drone.mp4", "drone v17 | heldout-903")
"""
from __future__ import annotations

import math
import os
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim.mujoco_backend import cell_center
from sim.skills import Sim
from sim.world import NOFLY_CEIL, T

W, H, FPS = 1280, 720, 30
MAX_SECONDS = 45          # longer missions are sped up to fit
WALL_2D = 2.5             # building height when the world is flat

TILE = {T.FLOOR: ".62 .63 .60 1", T.CHARGER: ".62 .63 .60 1", T.RUBBLE: ".52 .42 .32 1", T.STAIRS: ".70 .66 .58 1",
        T.WATER: ".22 .45 .70 .85", T.RAIL: ".45 .43 .42 1", T.AIR: ".62 .63 .60 1", T.NOFLY: ".62 .63 .60 1",
        T.WALL: ".40 .42 .45 1"}
ROBOT_RGB = [".95 .55 .10", ".15 .65 .95", ".85 .25 .55", ".35 .80 .35"]
ITEM_RGB = {"water": ".2 .5 .95", "radio": ".95 .8 .15", "supply_kit": ".9 .25 .2", "blanket": ".6 .35 .8"}


# ---- real robots from MuJoCo Menagerie (fetched by scripts/fetch_menagerie.sh) --------
MENAGERIE = os.path.join(os.path.dirname(__file__), "assets", "menagerie")
MODEL_FILE = {"skydio_x2": "skydio_x2/x2.xml", "unitree_go2": "unitree_go2/go2.xml",
              "boston_dynamics_spot": "boston_dynamics_spot/spot.xml", "anybotics_anymal_c": "anybotics_anymal_c/anymal_c.xml",
              "unitree_g1": "unitree_g1/g1.xml", "unitree_h1": "unitree_h1/h1.xml"}
DEFAULT_MODEL = {"drone": "skydio_x2", "quadruped": "unitree_go2", "humanoid": "unitree_g1"}
# standing base height and kinematic gait per model: (joint, standing angle, swing amp, lift amp, phase)
_Q4 = [(0.0, "FL"), (math.pi, "FR"), (math.pi, "RL"), (0.0, "RR")]
GAITS = {
    "unitree_go2": (0.27, 0.35, [(f"{l}_thigh_joint", .9, .3, 0, p) for p, l in _Q4] +
                    [(f"{l}_calf_joint", -1.8, 0, -.5, p) for p, l in _Q4]),
    "boston_dynamics_spot": (0.46, 0.6, [(f"{l}_hy", 1.04, .3, 0, p) for p, l in zip((0, math.pi, math.pi, 0), ("fl", "fr", "hl", "hr"))] +
                             [(f"{l}_kn", -1.8, 0, -.5, p) for p, l in zip((0, math.pi, math.pi, 0), ("fl", "fr", "hl", "hr"))]),
    "anybotics_anymal_c": (0.58, 0.55, [(f"{l}_HFE", s * .4, .3, 0, p) for p, l, s in
                                        zip((0, math.pi, math.pi, 0), ("LF", "RF", "LH", "RH"), (1, 1, -1, -1))] +
                           [(f"{l}_KFE", -s * .8, 0, -s * .5, p) for p, l, s in
                            zip((0, math.pi, math.pi, 0), ("LF", "RF", "LH", "RH"), (1, 1, -1, -1))]),
    "unitree_g1": (0.78, 0.5, [("left_hip_pitch_joint", -.1, .35, 0, 0), ("right_hip_pitch_joint", -.1, .35, 0, math.pi),
                               ("left_knee_joint", .25, 0, .6, 0), ("right_knee_joint", .25, 0, .6, math.pi),
                               ("left_ankle_pitch_joint", -.15, -.2, -.3, 0), ("right_ankle_pitch_joint", -.15, -.2, -.3, math.pi),
                               ("left_shoulder_pitch_joint", .2, -.35, 0, 0), ("right_shoulder_pitch_joint", .2, -.35, 0, math.pi),
                               ("left_elbow_joint", 1.0, 0, 0, 0), ("right_elbow_joint", 1.0, 0, 0, 0)]),
    "unitree_h1": (1.02, 0.65, [("left_hip_pitch", -.1, .35, 0, 0), ("right_hip_pitch", -.1, .35, 0, math.pi),
                                ("left_knee", .25, 0, .6, 0), ("right_knee", .25, 0, .6, math.pi),
                                ("left_ankle", -.15, -.2, -.3, 0), ("right_ankle", -.15, -.2, -.3, math.pi),
                                ("left_shoulder_pitch", .2, -.35, 0, 0), ("right_shoulder_pitch", .2, -.35, 0, math.pi),
                                ("left_elbow", 1.2, 0, 0, 0), ("right_elbow", 1.2, 0, 0, 0)]),
    "skydio_x2": (0.0, 1.0, []),
}


def _model_path(name: str) -> str:
    return os.path.join(MENAGERIE, MODEL_FILE[name]) if name in MODEL_FILE else name


def real_model(body) -> str | None:
    """Real model (Menagerie name or a user's model file) if present, else None (procedural model)."""
    name = body.model or DEFAULT_MODEL.get(body.kind, "")
    return name if name and os.path.exists(_model_path(name)) else None


def _auto_gait(path: str) -> tuple:
    """Gait for a model we have no hand-tuned entry for: stand height from geometry, then swing
    hip/thigh pitch joints and bend knee/calf joints, diagonal legs in antiphase."""
    import re
    from sim.robot_import import geom_extent, _visual_geoms
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    if m.nkey:
        d.qpos[:] = m.key_qpos[0]
    if m.njnt and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[:3] = 0
    mujoco.mj_forward(m, d)
    lo, hi = geom_extent(m, d, _visual_geoms(m))
    spec = []
    for j in range(m.njnt):
        name = m.joint(j).name
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        x, y = d.xanchor[j][:2]
        off = 0.0 if x * y >= 0 else math.pi
        stand = float(d.qpos[m.jnt_qposadr[j]])
        if re.search(r"(thigh|hip_pitch|hfe|_hy$|upper_leg)", name, re.I):
            spec.append((name, stand, .3, 0, off))
        elif re.search(r"(calf|knee|kfe|_kn$|lower_leg|shin)", name, re.I):
            spec.append((name, stand, 0, -.4 if stand < 0 else .4, off))
    return (-float(lo[2]), 0.5, spec)


def _attach(spec, rid: str, name: str):
    child = mujoco.MjSpec.from_file(_model_path(name))
    base = os.path.dirname(_model_path(name))
    child.meshdir = os.path.join(base, child.meshdir or "")
    child.texturedir = os.path.join(base, child.texturedir or "")
    for k in list(child.keys):
        child.delete(k)  # keyframes do not survive attachment into a bigger model
    root = child.worldbody.first_body()
    name0 = root.name
    spec.worldbody.add_frame().attach_body(root, f"{rid}/", "")
    return name0


# ---- procedural robot models -------------------------------------------------------
def _hinge(name: str, axis: str) -> str:
    return f'<joint name="{name}" type="hinge" axis="{axis}"/>'


def _quad(rid: str, c: str, s: float = 1.0, wheels: bool = False) -> tuple[str, list]:
    a = 0.20 * s
    parts = [f'<geom type="box" size="{.09*s} {.07*s} {.035*s}" rgba="{c} 1"/>',
             f'<geom type="sphere" size="{.04*s}" pos="{.08*s} 0 0" rgba=".1 .1 .12 1"/>',  # gimbal camera
             f'<geom type="box" size="{.06*s} {.05*s} {.012*s}" pos="0 0 {.045*s}" rgba=".15 .15 .17 1"/>']  # battery
    anim = []
    for i, (x, y) in enumerate(((a, a), (-a, a), (-a, -a), (a, -a))):
        parts.append(f'<geom type="capsule" fromto="0 0 0 {x} {y} 0" size="{.012*s}" rgba=".18 .18 .2 1"/>')
        parts.append(f'<geom type="cylinder" size="{.022*s} {.025*s}" pos="{x} {y} {.02*s}" rgba=".1 .1 .1 1"/>')
        parts.append(f'<body pos="{x} {y} {.05*s}">{_hinge(f"{rid}_rot{i}", "0 0 1")}'
                     f'<geom type="box" size="{.13*s} {.012*s} {.002}" rgba=".1 .1 .1 .9"/>'
                     f'<geom type="cylinder" size="{.13*s} .001" rgba=".7 .7 .75 .18"/></body>')
        anim.append((f"{rid}_rot{i}", "rotor", 1 if i % 2 else -1))
    for y in (-.08 * s, .08 * s):  # landing skids
        parts.append(f'<geom type="capsule" fromto="{-.1*s} {y} {-.1*s} {.1*s} {y} {-.1*s}" size="{.008*s}" rgba=".2 .2 .2 1"/>')
        parts.append(f'<geom type="capsule" fromto="0 {y} {-.1*s} 0 {y*.6} 0" size="{.006*s}" rgba=".2 .2 .2 1"/>')
    if wheels:
        for i, (x, y) in enumerate(((.13, .15), (-.13, .15), (-.13, -.15), (.13, -.15))):
            parts.append(f'<body pos="{x*s} {y*s} {-.06*s}">{_hinge(f"{rid}_wh{i}", "0 1 0")}'
                         f'<geom type="cylinder" size="{.06*s} {.02*s}" euler="90 0 0" rgba=".08 .08 .08 1"/>'
                         f'<geom type="box" size="{.04*s} {.021*s} {.008*s}" rgba=".5 .5 .5 1"/></body>')
            anim.append((f"{rid}_wh{i}", "wheel", .06 * s))
    return "".join(parts), anim


def _rover(rid: str, c: str) -> tuple[str, list]:
    r = .16
    parts = [f'<geom type="box" size=".38 .24 .1" pos="0 0 {r+.1}" rgba="{c} 1"/>',
             f'<geom type="box" size=".2 .18 .06" pos="-.05 0 {r+.26}" rgba=".2 .2 .22 1"/>',
             f'<geom type="cylinder" size=".02 .15" pos=".25 0 {r+.35}" rgba=".3 .3 .3 1"/>',  # mast
             f'<geom type="box" size=".05 .08 .04" pos=".27 0 {r+.52}" rgba=".1 .1 .1 1"/>',  # camera head
             f'<geom type="sphere" size=".025" pos=".32 .04 {r+.52}" rgba=".3 .7 1 1"/>',
             f'<geom type="sphere" size=".025" pos=".32 -.04 {r+.52}" rgba=".3 .7 1 1"/>']
    anim = []
    for i, (x, y) in enumerate(((.28, .3), (-.28, .3), (-.28, -.3), (.28, -.3))):
        parts.append(f'<body pos="{x} {y} {r}">{_hinge(f"{rid}_wh{i}", "0 1 0")}'
                     f'<geom type="cylinder" size="{r} .06" euler="90 0 0" rgba=".08 .08 .08 1"/>'
                     f'<geom type="box" size="{r*.7} .061 .03" rgba=".55 .55 .55 1"/>'
                     f'<geom type="box" size=".03 .061 {r*.7}" rgba=".55 .55 .55 1"/></body>')
        anim.append((f"{rid}_wh{i}", "wheel", r))
    return "".join(parts), anim


def _tracked(rid: str, c: str) -> tuple[str, list]:
    parts = [f'<geom type="box" size=".42 .22 .13" pos="0 0 .33" rgba="{c} 1"/>',
             '<geom type="box" size=".18 .16 .08" pos="-.05 0 .52" rgba=".25 .27 .22 1"/>',
             '<geom type="capsule" fromto=".1 0 .55 .45 0 .62" size=".035" rgba=".2 .2 .2 1"/>']  # arm
    anim = []
    for side in (-1, 1):
        y = side * .3
        parts.append(f'<geom type="capsule" fromto="-.42 {y} .17 .42 {y} .17" size=".16" rgba=".12 .12 .12 1"/>')
        for i, x in enumerate((-.3, -.1, .1, .3)):
            n = f"{rid}_rw{side}{i}"
            parts.append(f'<body pos="{x} {y + side*.17} .17">{_hinge(n, "0 1 0")}'
                         f'<geom type="cylinder" size=".1 .015" euler="90 0 0" rgba=".35 .35 .35 1"/>'
                         f'<geom type="box" size=".07 .016 .015" rgba=".6 .6 .6 1"/></body>')
            anim.append((n, "wheel", .1))
    return "".join(parts), anim


def _legged(rid: str, c: str, n_legs: int) -> tuple[str, list]:
    hip_z = .42 if n_legs == 4 else .3
    if n_legs == 4:
        parts = [f'<geom type="box" size=".34 .13 .08" pos="0 0 {hip_z}" rgba="{c} 1"/>',
                 f'<geom type="box" size=".08 .1 .06" pos=".4 0 {hip_z+.03}" rgba=".15 .15 .17 1"/>',  # head
                 f'<geom type="sphere" size=".025" pos=".48 .04 {hip_z+.04}" rgba=".3 .7 1 1"/>',
                 f'<geom type="sphere" size=".025" pos=".48 -.04 {hip_z+.04}" rgba=".3 .7 1 1"/>']
        mounts = [(.27, .15), (.27, -.15), (-.27, .15), (-.27, -.15)]
        phase = [0, math.pi, math.pi, 0]  # trot: diagonal pairs
        thigh, shin = .22, .22
    else:
        parts = [f'<geom type="ellipsoid" size=".3 .18 .07" pos="0 0 {hip_z}" rgba="{c} 1"/>',
                 f'<geom type="sphere" size=".07" pos=".3 0 {hip_z+.02}" rgba=".15 .15 .17 1"/>']
        mounts = [(.2, .17), (0, .2), (-.2, .17), (.2, -.17), (0, -.2), (-.2, -.17)]
        phase = [0, math.pi, 0, math.pi, 0, math.pi]  # tripod gait
        thigh, shin = .18, .3
    anim = []
    for i, (x, y) in enumerate(mounts):
        side = 1 if y > 0 else -1
        hip, knee = f"{rid}_hip{i}", f"{rid}_knee{i}"
        if n_legs == 4:  # sagittal legs: hip pitch + knee pitch
            leg = (f'<body pos="{x} {y} {hip_z}">{_hinge(hip, "0 1 0")}'
                   f'<geom type="capsule" fromto="0 0 0 0 0 {-thigh}" size=".035" rgba=".2 .2 .22 1"/>'
                   f'<body pos="0 0 {-thigh}">{_hinge(knee, "0 1 0")}'
                   f'<geom type="capsule" fromto="0 0 0 0 0 {-shin}" size=".025" rgba=".3 .3 .32 1"/>'
                   f'<geom type="sphere" size=".035" pos="0 0 {-shin}" rgba=".1 .1 .1 1"/></body></body>')
        else:  # spider legs: hip yaw + knee lift, splayed outward
            leg = (f'<body pos="{x} {y} {hip_z}">{_hinge(hip, "0 0 1")}'
                   f'<geom type="capsule" fromto="0 0 0 0 {side*thigh} .08" size=".025" rgba=".2 .2 .22 1"/>'
                   f'<body pos="0 {side*thigh} .08">{_hinge(knee, f"{side} 0 0")}'
                   f'<geom type="capsule" fromto="0 0 0 0 {side*.08} {-hip_z-.08}" size=".02" rgba=".3 .3 .32 1"/>'
                   f'</body></body>')
        parts.append(leg)
        anim.append((hip, "hip4" if n_legs == 4 else "hip6", phase[i]))
        anim.append((knee, "knee4" if n_legs == 4 else "knee6", phase[i]))
    return "".join(parts), anim


def _railcart(rid: str, c: str) -> tuple[str, list]:
    parts = [f'<geom type="box" size=".6 .3 .18" pos="0 0 .38" rgba="{c} 1"/>',
             '<geom type="box" size=".62 .32 .03" pos="0 0 .2" rgba=".2 .2 .2 1"/>',
             '<geom type="box" size=".5 .26 .04" pos="0 0 .58" rgba=".25 .25 .27 1"/>',
             '<geom type="sphere" size=".04" pos=".62 0 .45" rgba="1 .9 .5 1"/>']  # headlight
    anim = []
    for i, (x, y) in enumerate(((.4, .2), (-.4, .2), (-.4, -.2), (.4, -.2))):
        parts.append(f'<body pos="{x} {y} .1">{_hinge(f"{rid}_wh{i}", "0 1 0")}'
                     f'<geom type="cylinder" size=".1 .03" euler="90 0 0" rgba=".3 .3 .3 1"/>'
                     f'<geom type="box" size=".07 .031 .015" rgba=".7 .7 .7 1"/></body>')
        anim.append((f"{rid}_wh{i}", "wheel", .1))
    return "".join(parts), anim


def _fixed_wing(rid: str, c: str) -> tuple[str, list]:
    parts = [f'<geom type="capsule" fromto="-.55 0 0 .45 0 0" size=".07" rgba="{c} 1"/>',
             '<geom type="sphere" size=".06" pos=".42 0 .04" rgba=".2 .4 .6 .8"/>',  # canopy
             '<geom type="box" size=".14 .8 .012" pos=".02 0 .03" rgba=".92 .92 .94 1"/>',  # wing
             '<geom type="box" size=".07 .28 .008" pos="-.55 0 .02" rgba=".92 .92 .94 1"/>',  # tailplane
             '<geom type="box" size=".07 .008 .14" pos="-.55 0 .12" rgba="' + c + ' 1"/>']  # fin
    anim = []
    for i, y in enumerate((-.42, .42)):  # VTOL booms with fore/aft lift rotors
        parts.append(f'<geom type="capsule" fromto="-.35 {y} .02 .38 {y} .02" size=".022" rgba=".2 .2 .22 1"/>')
        for j, x in enumerate((-.32, .35)):
            n = f"{rid}_rot{i}{j}"
            parts.append(f'<body pos="{x} {y} .07">{_hinge(n, "0 0 1")}'
                         f'<geom type="box" size=".15 .013 .002" rgba=".1 .1 .1 .9"/>'
                         f'<geom type="cylinder" size=".15 .001" rgba=".7 .7 .75 .15"/></body>')
            anim.append((n, "vtol", 1 if (i + j) % 2 else -1))
    parts.append(f'<body pos="-.6 0 0">{_hinge(f"{rid}_push", "1 0 0")}'
                 '<geom type="box" size=".005 .14 .015" rgba=".1 .1 .1 1"/></body>')
    anim.append((f"{rid}_push", "pusher", 1))
    return "".join(parts), anim


def robot_xml(rid: str, kind: str, idx: int) -> tuple[str, list]:
    c = ROBOT_RGB[idx % len(ROBOT_RGB)]
    if kind == "drone":
        return _quad(rid, c, 1.5)
    if kind == "hybrid":
        return _quad(rid, c, 1.7, wheels=True)
    if kind == "rover":
        return _rover(rid, c)
    if kind == "tracked":
        return _tracked(rid, c)
    if kind in ("quadruped", "hexapod"):
        return _legged(rid, c, 4 if kind == "quadruped" else 6)
    if kind == "railcart":
        return _railcart(rid, c)
    if kind == "fixed_wing_vtol":
        return _fixed_wing(rid, c)
    return _rover(rid, c)


# ---- city scene ----------------------------------------------------------------
def _ground_layer(sim: Sim) -> np.ndarray:
    g = sim.world.grid
    if sim.world.dims == 1:
        return g[:, None]
    return g if sim.world.dims == 2 else g[:, :, 0]


def scene_xml(sim: Sim, kinds: dict[str, str], real: frozenset = frozenset()) -> tuple[str, dict]:
    ground = _ground_layer(sim)
    w, h = ground.shape
    rng = np.random.default_rng(sim.world.seed)
    geoms = []
    for x in range(w):  # run-length tiles per row
        y = 0
        while y < h:
            t = T(int(ground[x, y]))
            y1 = y
            while y1 + 1 < h and T(int(ground[x, y1 + 1])) == t:
                y1 += 1
            n = y1 - y + 1
            z = -.06 if t == T.WATER else -.02
            geoms.append(f'<geom type="box" size=".5 {n/2} .02" pos="{x+.5} {y+n/2} {z}" rgba="{TILE.get(t, TILE[T.FLOOR])}"/>')
            y = y1 + 1
    # buildings
    if sim.world.dims == 3:
        heights = (sim.world.grid == T.WALL).sum(axis=2)
    else:
        heights = (ground == T.WALL) * WALL_2D
    tone = {}
    for x, y in np.argwhere(heights > 0):
        hh = float(heights[x, y])
        key = (x // 8, y // 8)
        tone.setdefault(key, rng.uniform(.55, .78) * np.array([1.0, .97, .92]) + rng.uniform(-.04, .04, size=3))
        r, g_, b = tone[key]
        geoms.append(f'<geom type="box" size=".5 .5 {hh/2}" pos="{x+.5} {y+.5} {hh/2}" rgba="{r:.2f} {g_:.2f} {b:.2f} 1"/>')
        geoms.append(f'<geom type="box" size=".46 .46 .03" pos="{x+.5} {y+.5} {hh+.02}" rgba=".22 .23 .25 1"/>')
    for x, y in np.argwhere(ground == T.RUBBLE):
        for _ in range(3):
            dx, dy, sz = rng.uniform(-.35, .35), rng.uniform(-.35, .35), rng.uniform(.06, .14)
            geoms.append(f'<geom type="box" size="{sz} {sz*.8} {sz*.6}" pos="{x+.5+dx} {y+.5+dy} {sz*.4}" '
                         f'euler="{rng.uniform(0, 40):.0f} {rng.uniform(0, 40):.0f} {rng.uniform(0, 90):.0f}" rgba=".45 .36 .28 1"/>')
    for x, y in np.argwhere(ground == T.STAIRS):
        for k in range(3):
            geoms.append(f'<geom type="box" size=".45 {.15} {.04*(k+1)}" pos="{x+.5} {y+.2+.3*k} {.04*(k+1)}" rgba=".72 .68 .6 1"/>')
    for x, y in np.argwhere(ground == T.RAIL):
        for oy in (-.18, .18):
            geoms.append(f'<geom type="box" size=".5 .02 .03" pos="{x+.5} {y+.5+oy} .03" rgba=".55 .55 .6 1"/>')
        geoms.append(f'<geom type="box" size=".06 .3 .015" pos="{x+.5} {y+.5} .01" rgba=".35 .25 .18 1"/>')
    for c in sim.world.chargers:
        x, y = c[0], c[1] if len(c) > 1 else 0
        geoms.append(f'<geom type="cylinder" size=".42 .01" pos="{x+.5} {y+.5} .01" rgba=".1 .75 .35 1"/>')
        geoms.append(f'<geom type="cylinder" size=".04 .4" pos="{x+.9} {y+.9} .4" rgba=".1 .9 .4 1"/>')
        geoms.append(f'<geom type="sphere" size=".07" pos="{x+.9} {y+.9} .82" rgba=".3 1 .5 1"/>')
    if sim.world.dims == 3:
        nf = np.argwhere((sim.world.grid[:, :, 1:] == T.NOFLY).any(axis=2))
        if len(nf):
            (x0, y0), (x1, y1) = nf.min(0), nf.max(0) + 1
            geoms.append(f'<geom type="box" size="{(x1-x0)/2} {(y1-y0)/2} {(NOFLY_CEIL-1)/2}" '
                         f'pos="{(x0+x1)/2} {(y0+y1)/2} {(NOFLY_CEIL+1)/2}" rgba="1 .15 .1 .13"/>')
    bodies, anims = [], {}
    for i, (rid, kind) in enumerate(kinds.items()):
        if rid in real:  # attached from MuJoCo Menagerie after compile
            anims[rid] = []
            continue
        xml, anim = robot_xml(rid, kind, i)
        bodies.append(f'<body name="{rid}" mocap="true">{xml}</body>')
        anims[rid] = anim
    for it in sim.items.values():
        rgb = ITEM_RGB.get(it.id, ".9 .9 .9")
        bodies.append(f'<body name="item_{it.id}" mocap="true">'
                      f'<geom type="box" size=".12 .12 .1" pos="0 0 .1" rgba="{rgb} 1"/>'
                      f'<geom type="box" size=".125 .03 .101" pos="0 0 .1" rgba=".95 .95 .95 1"/></body>')
    cx, cy = w / 2, h / 2
    xml = f"""
<mujoco model="morph-city">
  <visual>
    <global offwidth="{W}" offheight="{H}" fovy="40"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight ambient=".35 .35 .38" diffuse=".45 .45 .45" specular=".1 .1 .1"/>
    <map znear=".02" zfar="400" haze=".25" fogstart="40" fogend="140"/>
    <rgba haze=".75 .83 .92 1" fog=".75 .83 .92 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".55 .72 .92" rgb2=".95 .97 1" width="512" height="512"/>
    <texture name="grass" type="2d" builtin="checker" rgb1=".36 .48 .30" rgb2=".33 .45 .28" width="512" height="512"/>
    <material name="grass" texture="grass" texrepeat="60 60"/>
  </asset>
  <worldbody>
    <light directional="true" pos="{cx} {cy} 60" dir="-.35 -.25 -1" diffuse=".75 .72 .68" castshadow="true"/>
    <geom type="plane" size="{max(w, h)*2} {max(w, h)*2} .1" pos="{cx} {cy} -.045" material="grass"/>
    {''.join(geoms)}
    {''.join(bodies)}
  </worldbody>
</mujoco>"""
    return xml, anims


# ---- replay ---------------------------------------------------------------------
def _font(size: int):
    for f in ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _hud(img: np.ndarray, title: str, f: dict, speed: float, fonts) -> np.ndarray:
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    d.rectangle([0, 0, W, 52], fill=(10, 14, 22, 170))
    d.text((18, 13), title, font=fonts[0], fill=(240, 244, 250))
    d.rectangle([0, H - 44, W, H], fill=(10, 14, 22, 150))
    status = f"t={f['t']:6.1f}s  x{speed:.0f}  {f['robot']} [{f['mode']}]"
    if f["carrying"]:
        status += f"  carrying {', '.join(f['carrying'])}"
    d.text((18, H - 34), status, font=fonts[1], fill=(220, 226, 236))
    b = max(0.0, min(1.0, f["battery"]))
    x0 = W - 260
    d.text((x0 - 80, H - 34), "battery", font=fonts[1], fill=(220, 226, 236))
    d.rectangle([x0, H - 32, x0 + 200, H - 14], outline=(220, 226, 236), width=2)
    col = (60, 210, 110) if b > .4 else (240, 190, 40) if b > .2 else (235, 70, 60)
    d.rectangle([x0 + 3, H - 29, x0 + 3 + int(194 * b), H - 17], fill=col)
    if f.get("label"):
        tw = d.textlength(f["label"].upper(), font=fonts[0])
        d.rounded_rectangle([W / 2 - tw / 2 - 16, 70, W / 2 + tw / 2 + 16, 112], 10, fill=(20, 120, 220, 200))
        d.text((W / 2 - tw / 2, 78), f["label"].upper(), font=fonts[0], fill=(255, 255, 255))
    d.text((W - 300, 16), "MuJoCo rigid-body replay", font=fonts[1], fill=(170, 200, 255))
    return np.asarray(im)


def render3d(sim: Sim, path: str, title: str = "", follow: str | None = None, panel=None,
             max_seconds: float = MAX_SECONDS, compose=None, hud: bool = True) -> str:
    """Render `sim.trace` (from a MujocoSim run) to an MP4 at `path`.

    `panel(frame, progress) -> PIL.Image` (height H) is drawn to the right of the 3D view."""
    trace = getattr(sim, "trace", None)
    if trace is None:
        raise ValueError("no MuJoCo trace: run the program with physics='mujoco'")
    if not trace:  # the program never moved a robot (e.g. its first skill failed): show it standing still
        errs = [e["reason"] for e in sim.events if e["kind"] == "skill_error"]
        label = "no movement: " + (errs[0] if errs else "empty plan")
        for rid, r in sim.robots.items():
            p = list(cell_center(r.pos)) if len(r.pos) == 3 else [r.pos[0] + .5, (r.pos[1] if len(r.pos) > 1 else 0) + .5, 0.0]
            trace = [{"t": round(i * 0.04, 3), "robot": rid, "kind": r.body.kind, "mode": r.body.modes[0], "pos": p,
                      "quat": [1, 0, 0, 0], "act": 0.0, "battery": r.battery / r.body.battery_wh,
                      "carrying": list(r.carrying), "label": label} for i in range(75)]
            break
    kinds = {rid: r.body.kind for rid, r in sim.robots.items()}
    real = {rid: m for rid, r in sim.robots.items() if (m := real_model(r.body))}
    xml, anims = scene_xml(sim, kinds, frozenset(real))
    spec = mujoco.MjSpec.from_string(xml)
    roots = {rid: _attach(spec, rid, m) for rid, m in real.items()}
    model = spec.compile()
    data = mujoco.MjData(model)
    mocap = {rid: model.body_mocapid[model.body(rid).id] for rid in kinds if rid not in real}
    free = {rid: model.jnt_qposadr[model.body_jntadr[model.body(f"{rid}/{root}").id]] for rid, root in roots.items()}
    gait = {}
    for m in set(real.values()) - set(GAITS):
        GAITS[m] = _auto_gait(_model_path(m))
    for rid, m in real.items():
        gait[rid] = []
        for n, stand, swing, lift, off in GAITS[m][2]:
            try:
                gait[rid].append((model.jnt_qposadr[model.joint(f"{rid}/{n}").id], stand, swing, lift, off))
            except KeyError:
                pass
    items = {it.id: model.body_mocapid[model.body(f"item_{it.id}").id] for it in sim.items.values()}
    item_pos = {}
    for it_id, it in _initial_items(sim, trace).items():
        item_pos[it_id] = it
    jadr = {n: model.jnt_qposadr[model.joint(n).id] for a in anims.values() for n, *_ in a}
    last = {}
    for rid, r in sim.robots.items():  # start poses
        first = next((f for f in trace if f["robot"] == rid), None)
        last[rid] = first or {"pos": list(cell_center(r.pos)) if len(r.pos) == 3 else [r.pos[0] + .5, (r.pos[1] if len(r.pos) > 1 else 0) + .5, 0],
                              "quat": [1, 0, 0, 0], "act": 0, "mode": r.body.modes[0], "carrying": []}
    duration = trace[-1]["t"]
    stride = max(1, math.ceil(len(trace) / (max_seconds * FPS)))
    pw = panel(trace[0], 0.0).width if panel else 0
    size = compose(np.zeros((H, W, 3), np.uint8), trace[0], 0, 0.0).size if compose else (W + pw, H)
    speed = stride * (duration / max(len(trace), 1)) * FPS
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    fonts = (_font(22), _font(17))
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-crf", "20", "-movflags", "+faststart", path], stdin=subprocess.PIPE)
    phase = {rid: 0.0 for rid in kinds}
    angle = {n: 0.0 for n in jadr}
    look = None
    prev_t = 0.0
    try:
        for k, f in enumerate(trace):
            rid = f["robot"]
            dt = f["t"] - prev_t
            prev_t = f["t"]
            last[rid] = f
            # items follow their carrier; dropped items stay where they were put down
            for it_id in items:
                carrier = next((r for r, lf in last.items() if it_id in lf.get("carrying", [])), None)
                if carrier:
                    p = np.array(last[carrier]["pos"])
                    item_pos[it_id] = p + ([0, 0, -.32] if p[2] > .3 else [0, 0, .35])
                elif item_pos[it_id][2] > 0.05 and not any(it_id in lf.get("carrying", []) for lf in last.values()):
                    item_pos[it_id] = np.array([item_pos[it_id][0], item_pos[it_id][1], 0.0])
            # animate parts of the robot this frame belongs to
            for n, kind, arg in anims[rid]:
                act, spd = f["act"], f["act"]
                if kind == "rotor":
                    angle[n] += arg * (40 + 60 * act) * dt if act > 0.02 else 0
                elif kind == "vtol":
                    lift = 1.0 if f["mode"] == "fixed_wing" and f["pos"][2] > .3 and spd < 2.0 else 0.0
                    angle[n] += arg * 90 * lift * dt
                elif kind == "pusher":
                    angle[n] += 120 * dt if spd > 1.0 else 0
                elif kind == "wheel":
                    angle[n] += spd * dt / arg
            if rid in real:
                phase[rid] += f["act"] * dt * 2 * math.pi / GAITS[real[rid]][1] if f["mode"] != "flight" else 0
            else:
                phase[rid] += f["act"] * dt * 7.0
            ph = phase[rid]
            for n, kind, off in anims[rid]:
                if kind == "hip4":
                    angle[n] = 0.45 * math.sin(ph + off)
                elif kind == "knee4":
                    angle[n] = -0.15 - 0.6 * max(0.0, math.sin(ph + off + math.pi / 2))
                elif kind == "hip6":
                    angle[n] = 0.35 * math.sin(ph + off)
                elif kind == "knee6":
                    angle[n] = -0.35 * max(0.0, math.cos(ph + off))
            final = k == len(trace) - 1
            if k % stride and not final:
                continue
            for r_, lf in last.items():
                if r_ in real:
                    a = free[r_]
                    data.qpos[a:a + 3] = np.array(lf["pos"]) + [0, 0, GAITS[real[r_]][0]]
                    data.qpos[a + 3:a + 7] = lf["quat"]
                    amp = min(1.0, lf["act"] / 0.4) if lf["mode"] != "flight" else 0.0
                    for adr, stand, swing, lift, off in gait[r_]:
                        x = phase[r_] + off
                        data.qpos[adr] = stand + amp * (swing * math.sin(x) + lift * max(0.0, math.sin(x + math.pi / 2)))
                else:
                    data.mocap_pos[mocap[r_]] = lf["pos"]
                    data.mocap_quat[mocap[r_]] = lf["quat"]
            for it_id, mid in items.items():
                data.mocap_pos[mid] = item_pos[it_id]
            for n, adr in jadr.items():
                data.qpos[adr] = angle[n]
            mujoco.mj_forward(model, data)
            target = np.array(last[follow or rid]["pos"])
            look = target if look is None else look + (target - look) * min(1.0, 0.08 * stride)
            air = target[2] > 0.5
            cam.lookat[:] = look
            cam.distance = 13.0 if air else 6.5
            cam.elevation = -80 if air else -42  # steep when airborne: buildings never block the view
            cam.azimuth = 135 + 12 * math.sin(f["t"] / 25)
            renderer.update_scene(data, cam)
            img = _hud(renderer.render(), title, f, speed, fonts) if hud else renderer.render()
            if compose:
                img = np.asarray(compose(img, f, k, 1.0 if final else k / len(trace)).convert("RGB"))
            elif panel:
                img = np.hstack([img, np.asarray(panel(f, 1.0 if final else k / len(trace)).convert("RGB"))])
            buf = np.ascontiguousarray(img).tobytes()
            for _ in range(int(FPS * 2.5) if final and (panel or compose) else 1):  # hold on the result
                ff.stdin.write(buf)
    finally:
        ff.stdin.close()
        ff.wait()
        renderer.close()
    return path


def _initial_items(sim: Sim, trace: list[dict]) -> dict:
    """Item start positions in metres (sim.items holds the *final* state after the run)."""
    out = {}
    for it in sim.items.values():
        p = sim.__dict__.get("items0", {}).get(it.id) or it.pos or (0, 0)
        out[it.id] = np.array([p[0] + .5, (p[1] if len(p) > 1 else 0) + .5, 0.0])
    return out
