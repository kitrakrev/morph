"""MuJoCo rotorcraft backend: same skill interface, real rigid-body execution."""
import pytest

pytest.importorskip("mujoco")

from sim import tier2, verifier  # noqa: E402
from sim.mission import build_sim  # noqa: E402
from sim.missions import BODIES, make_mission  # noqa: E402
from sim.mujoco_backend import MujocoSim  # noqa: E402
from sim.skills import Sim  # noqa: E402


def _drone_at(pos, physics="mujoco"):
    return build_sim({"physics": physics, "terrain": {"dims": 3, "seed": 7, "difficulty": "easy"},
                      "robots": [{"id": "d", "structure": BODIES["drone"], "position": pos}],
                      "items": [], "goal": {}})


def _program(m, alt=2):
    rid = m["robots"][0]["id"]
    chargers = build_sim(m).world.chargers

    def near(p):
        return list(min(chargers, key=lambda c: abs(c[0] - p[0]) + abs(c[1] - p[1])))

    def fly(p):
        return {"robot": rid, "skill": "fly_to", "args": {"pos": list(p), "alt": alt, "avoid_nofly": True}}
    g = m["goal"]["deliver"]
    it = next(i for i in m["items"] if i["id"] == g["item"])
    charge = {"robot": rid, "skill": "charge", "args": {}}
    return [fly(near(it["pos"])), charge, fly(it["pos"]), {"robot": rid, "skill": "pick", "args": {"item": g["item"]}},
            fly(near(g["to"])), charge, fly(g["to"]), {"robot": rid, "skill": "drop", "args": {"item": g["item"]}}]


def test_backend_selection():
    assert type(_drone_at([0, 0, 0], "grid")) is Sim
    assert isinstance(_drone_at([0, 0, 0]), MujocoSim)


def test_flight_tracks_and_matches_grid_energy():
    sims = {be: _drone_at([0, 0, 0], be) for be in ("grid", "mujoco")}
    for sim in sims.values():
        assert sim.fly_to("d", [0, 1, 0], alt=3)["ok"]
    mj = sims["mujoco"]
    assert not [e for e in mj.events if e["kind"] == "collision"]
    assert mj._quads["d"].peak_error < 0.4
    g, m = sims["grid"].robots["d"].drained_wh, mj.robots["d"].drained_wh
    assert abs(m - g) / g < 0.35


def test_clone_drops_mujoco_state():
    sim = _drone_at([0, 0, 0])
    sim.fly_to("d", [0, 1, 0], alt=2)
    c = sim.clone()
    assert isinstance(c, MujocoSim) and c._quads == {} and sim._quads
    assert c.fly_to("d", [0, 0, 0], alt=2)["ok"]


@pytest.mark.parametrize("seed", [903, 907])
def test_heldout_delivery_succeeds_on_real_dynamics(seed):
    m = make_mission("drone", seed, "heldout")
    prog = _program(m)
    grid = verifier.score(m, tier2.simulate(m, prog))
    mj = verifier.score(m, tier2.simulate({**m, "physics": "mujoco"}, prog))
    assert grid["success"] and mj["success"], mj
    assert abs(mj["energy_wh"] - grid["energy_wh"]) / grid["energy_wh"] < 0.2
