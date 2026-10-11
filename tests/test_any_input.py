"""Any robot (model file), any map (ASCII / JSON / PNG, 1D-3D), any goal (composable tasks)."""
import os

import pytest

from engine.harness import UnsupportedBody, bootstrap
from engine.planner import compile_program
from sim import spec, tier2, verifier
from sim.missions import mission_set
from sim.world import T, world_from_spec

MEN = "sim/assets/menagerie"
need_men = pytest.mark.skipif(not os.path.isdir(MEN), reason="run scripts/fetch_menagerie.sh")


@need_men
@pytest.mark.parametrize("path,loco", [("skydio_x2/x2.xml", {"rotors": 4}), ("unitree_go2/go2.xml", {"legs": 4}),
                                       ("boston_dynamics_spot/spot.xml", {"legs": 4}), ("unitree_g1/g1.xml", {"legs": 2})])
def test_locomotion_from_model_file(path, loco):
    from sim.robot_import import structure_from_model
    assert structure_from_model(f"{MEN}/{path}")["locomotion"] == loco


@need_men
def test_immobile_model_is_refused():
    with pytest.raises(UnsupportedBody):
        bootstrap({"model_file": f"{MEN}/franka_emika_panda/panda.xml", "battery_wh": 100})


def test_maps_any_dimension():
    w3 = world_from_spec({"map": ["C..##", ".3...", "....."], "nofly": [[3, 1, 5, 3]]})
    assert w3.dims == 3 and w3.grid[1, 1, 2] == T.WALL and w3.grid[3, 1, 1] == T.NOFLY
    assert world_from_spec({"map": ["C.#", "..~"]}).dims == 2
    assert world_from_spec({"map": ["==C=="], "dims": 1}).dims == 1


GOALS = {
    "inspect+home": [{"type": "inspect", "pos": [4, 0]}, {"type": "return_home"}],
    "patrol": [{"type": "patrol", "points": [[4, 0], [4, 3], [0, 0]], "laps": 2}],
    "visit": [{"type": "visit", "pos": [3, 3]}],
    "survey": [{"type": "survey", "area": [0, 0, 5, 4], "coverage": 0.9}],
    "avoid": [{"type": "visit", "pos": [4, 3]}, {"type": "avoid", "area": [10, 10, 12, 12]}],
}


@pytest.mark.parametrize("name", list(GOALS))
def test_goal_types_checked_from_physics(name):
    m = {"physics": "grid", "terrain": {"map": ["C....", ".....", "..#..", "....."]},
         "robots": [{"id": "r", "structure": {"locomotion": {"wheels": 4}, "mass_kg": 20, "battery_wh": 100,
                                              "sensors": ["lidar_2m"]}, "position": [0, 3]}],
         "items": [], "goal": {"tasks": GOALS[name]}}
    tpl = [{"foreach": "$TARGETS", "do": [{"skill": "move_to", "args": {"pos": "$TARGET"}}, {"skill": "scan"}]},
           {"skill": "move_to", "args": {"pos": "$HOME"}}]
    s = verifier.score(m, tier2.simulate(m, compile_program(tpl, m, {"tool_specs": {}})))
    assert s["success"], s["task_status"]
    idle = verifier.score(m, tier2.simulate(m, []))  # doing nothing must not satisfy an active goal
    assert not idle["success"] or name == "avoid"


@need_men
def test_spec_registers_body_with_disjoint_splits():
    s = spec.load("specs/go2-plant-inspection.json")
    tr, ho = mission_set(s["name"], "train", 3), mission_set(s["name"], "heldout", 3)
    assert len(tr) == len(ho) == 3 and not {m["id"] for m in tr} & {m["id"] for m in ho}
