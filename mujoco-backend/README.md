# Morph — MuJoCo backend

The same self-evolving harness (planner, workflow graph, locked verifier, gate, Atlas memory) running on
**MuJoCo rigid-body physics** instead of the closed-form grid model. Run everything from this folder.

```bash
cd mujoco-backend
scripts/fetch_menagerie.sh                       # real robot models (~250 MB, gitignored)
PYTHONPATH=. uv run --with pytest pytest tests   # 23 tests
```

## What is different from `simple-backend/`

| | |
|---|---|
| Physics | `sim/mujoco_backend.py` — every skill move is executed in MuJoCo. Rotorcraft: 4-rotor quad with a geometric controller, momentum-theory rotor power, building collisions, altitude gusts. Ground (wheels, tracks, legs, rail) and fixed-wing VTOL: driven rigid bodies with terrain rolling resistance or wing lift/drag. Energy calibrated so a steady unloaded floor cell matches the grid model. `MORPH_PHYSICS=grid` switches back. |
| Robots | Any robot as a MuJoCo model file: `{"model_file": ".../go2.xml", "battery_wh": 230}` — locomotion, mass and size are read from actuators and kinematics (`sim/robot_import.py`). Immobile models (an arm) are refused. Added bodies: Spot, ANYmal C, Unitree G1/H1. |
| Maps | `terrain: {"map": [...]}` / `map_file` (ASCII, JSON, PNG height map), 1D/2D/3D, digits = building heights, `nofly` boxes (`sim/world.py`). |
| Goals | Composable tasks checked by the verifier from the physics log: deliver, visit, inspect, patrol, return_home, survey, avoid (`sim/goals.py`). Planner templates get `$TARGETS`, `$HOME` and `foreach`. |
| Specs | One JSON = robot + map + goal (`specs/*.json`); `python -m engine.loop --spec <file>` evolves on seeded train / regression / held-out variants. |
| Videos | `scripts/render_mujoco.py` (3D replay per body), `scripts/attempts_with_params_3d.py` (every attempt + inputs + harness diff + why-it-won card), `scripts/render_evolution.py`. Robots are Menagerie meshes with scripted gaits (not walking policies). |
| Memory | `engine/multimodal.py` — failure map picture + diagnosis embedded with `voyage-multimodal-3.5`; third `$rankFusion` pipeline (`failures_mm` index). |

## Results (MuJoCo physics, held-out missions, same planner model)

| Spec | v1 | evolved | violations |
|---|---|---|---|
| Unitree Go2 (model file) · custom plant map · inspect 2 points + return home | 0/12 | **12/12** (v3; v2 rejected: invented a non-primitive skill) | 0 |
| Skydio X2 (model file) · 3D custom map · survey area + return home | 0/6 train | **6/6 train, 4/4 regression** (v3; v2 rejected) | 0 |
| Unitree G1 (model file) · hard 2D city · deliver + return home | 6/6 train from bootstrap | — | 0 |
| Drone heldout-903 replayed (grid-evolved v17) | fails | delivers | 0 |
