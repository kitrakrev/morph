"""Import a robot from its MuJoCo model file (MJCF; URDF loads too). Fixed code.

The user hands the harness a model file plus what a model file cannot say
(battery, payload). Everything else is read from the model:

  * mass         - total body mass
  * size_m       - bounding box in the model's default pose
  * locomotion   - from the actuator / kinematic structure:
        rotors     actuators that push along a site's z axis (thrust sites)
        wheels     unlimited hinge joints on round geoms (cylinder / sphere / ellipsoid)
        legs       limb chains: leaf bodies with >= 2 hinge ancestors whose tips sit
                   near the ground (feet). Arms (tips high up) are not legs.
        fixed_wing a geom much wider than long and thin (a wing) plus thrust
  * a model with none of these is refused by the harness bootstrap (UnsupportedBody),
    exactly like a JSON structure with an unknown locomotion.

    structure = {"model_file": "sim/assets/menagerie/unitree_go2/go2.xml", "battery_wh": 230}
"""
from __future__ import annotations

import functools
import os

import mujoco
import numpy as np


def _load(path: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(path)


def _pose(m: mujoco.MjModel) -> mujoco.MjData:
    d = mujoco.MjData(m)
    if m.nkey:
        d.qpos[:] = m.key_qpos[0]
    mujoco.mj_forward(m, d)
    return d


def geom_extent(m: mujoco.MjModel, d: mujoco.MjData, geoms) -> tuple[np.ndarray, np.ndarray]:
    """World-frame AABB (lo, hi) of the given geoms in the current pose."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in geoms:
        c, half = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        R = d.geom_xmat[g].reshape(3, 3)
        center = d.geom_xpos[g] + R @ c
        ext = np.abs(R) @ half
        lo, hi = np.minimum(lo, center - ext), np.maximum(hi, center + ext)
    return lo, hi


def _visual_geoms(m: mujoco.MjModel):
    # prefer render geoms (group 0-2); fall back to all
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0 and m.geom_group[g] <= 2 and m.geom_rgba[g][3] > 0]
    return gs or [g for g in range(m.ngeom) if m.geom_bodyid[g] > 0]


def _children(m: mujoco.MjModel) -> dict[int, list[int]]:
    ch: dict[int, list[int]] = {b: [] for b in range(m.nbody)}
    for b in range(1, m.nbody):
        ch[int(m.body_parentid[b])].append(b)
    return ch


def _hinges_on_path(m: mujoco.MjModel, b: int) -> int:
    n = 0
    while b > 0:
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]):
            n += m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        b = int(m.body_parentid[b])
    return n


def analyse(path: str) -> dict:
    """Structural facts about a model; the basis for `structure_from_model`."""
    m = _load(path)
    d = _pose(m)
    geoms = _visual_geoms(m)
    lo, hi = geom_extent(m, d, geoms)
    height = float(hi[2] - lo[2])
    # rotors: actuators with a site transmission pushing along +z of the site
    rotors = sum(1 for a in range(m.nu) if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_SITE
                 and abs(m.actuator_gear[a][2]) > 0 and np.allclose(m.actuator_gear[a][:2], 0))
    # wheels: unlimited hinges whose body carries a round geom
    round_types = (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID)
    wheels = 0
    for j in range(m.njnt):
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or m.jnt_limited[j]:
            continue
        b = m.jnt_bodyid[j]
        gs = range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])
        if any(m.geom_type[g] in round_types for g in gs) or "wheel" in m.joint(j).name.lower():
            wheels += 1
    # legs: leaf bodies with >= 2 hinge ancestors whose lowest point is near the ground
    ch = _children(m)
    legs = 0
    for b in range(1, m.nbody):
        if ch[b] or _hinges_on_path(m, b) < 2:
            continue
        gs = [g for g in geoms if m.geom_bodyid[g] == b] or \
             [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        tip = geom_extent(m, d, gs)[0][2] if gs else d.xpos[b][2]
        if tip - lo[2] < 0.12 * max(height, 1e-3):
            legs += 1
    # wings: a very wide, flat geom
    wing = any(m.geom_size[g][1] > 4 * m.geom_size[g][0] and m.geom_size[g][2] < 0.1 * m.geom_size[g][1]
               for g in geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX)
    return {"mass_kg": float(m.body_subtreemass[0]), "size_m": [round(float(x), 3) for x in hi - lo],
            "rotors": rotors, "wheels": wheels, "legs": legs, "wing": wing,
            "actuators": int(m.nu), "joints": int(m.njnt), "name": os.path.splitext(os.path.basename(path))[0]}


@functools.lru_cache(maxsize=64)
def _analyse_cached(path: str) -> dict:
    return analyse(path)


def structure_from_model(path: str, battery_wh: float | None = None, payload_bay_kg: float | None = None,
                         sensors: list[str] | None = None) -> dict:
    a = _analyse_cached(os.path.abspath(path))
    loco: dict = {}
    if a["wing"] and a["rotors"]:
        loco["fixed_wing"] = 1
    elif a["rotors"] >= 3:
        loco["rotors"] = a["rotors"]
    if a["legs"] >= 2:
        loco["legs"] = a["legs"]
    elif a["wheels"] >= 2:
        loco["wheels"] = a["wheels"]
    mass = round(a["mass_kg"], 2)
    return {"locomotion": loco, "mass_kg": mass,
            "battery_wh": battery_wh if battery_wh is not None else round(20 * mass + 10, 1),
            "payload_bay_kg": payload_bay_kg if payload_bay_kg is not None else round(0.3 * mass, 2),
            "sensors": sensors or ["camera"], "size_m": a["size_m"], "model": os.path.abspath(path),
            "derived_from": {k: a[k] for k in ("name", "rotors", "wheels", "legs", "wing", "actuators", "joints")}}


def resolve(structure: dict) -> dict:
    """Fill a structure that names a `model_file` from the model itself; explicit keys win."""
    if not structure.get("model_file") or structure.get("locomotion"):
        return structure
    derived = structure_from_model(structure["model_file"], structure.get("battery_wh"),
                                   structure.get("payload_bay_kg"), structure.get("sensors"))
    return {**derived, **{k: v for k, v in structure.items() if v is not None}}
