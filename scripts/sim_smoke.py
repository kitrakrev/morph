import time
import numpy as np
from sim.world import make_world, T
from sim import tier1, tier2, verifier
from sim.render import render

def floor_cells(w):
    g = w.grid if w.dims == 2 else w.grid[:, :, 0]
    return [tuple(int(c) for c in p) for p in np.argwhere(g == T.FLOOR)]

# --- rover, 2D ---
w = make_world(2, seed=7); cells = floor_cells(w)
start, src, dst = cells[0], cells[len(cells)//3], cells[-1]
rover = {"id": "rover-1", "structure": {"locomotion": {"wheels": 4}, "mass_kg": 30, "battery_wh": 200,
         "payload_bay_kg": 10, "sensors": ["camera"]}, "position": list(start)}
m = {"terrain": {"dims": 2, "seed": 7}, "robots": [rover],
     "items": [{"id": "water", "weight_kg": 4, "pos": list(src)}],
     "goal": {"deliver": {"item": "water", "to": list(dst)}, "constraints": ["battery_reserve"], "deadline_steps": 400}}
prog = [{"robot": "rover-1", "skill": "move_to", "args": {"pos": list(src)}},
        {"robot": "rover-1", "skill": "pick", "args": {"item": "water"}},
        {"robot": "rover-1", "skill": "move_to", "args": {"pos": list(dst)}},
        {"robot": "rover-1", "skill": "drop", "args": {"item": "water"}}]
t0 = time.time(); t1 = tier1.check(m, prog); t1s = time.time() - t0
sim = tier2.simulate(m, prog); s = verifier.score(m, sim)
print("ROVER tier1", t1, f"{t1s*1000:.0f}ms"); print("ROVER verify", s)
render(sim, "out/rover.mp4", "rover v1"); print("rendered out/rover.mp4", len(sim.frames), "frames")

# --- drone, 3D: radio 0.8kg, naive alt=2 route ---
w3 = make_world(3, seed=7); c3 = floor_cells(w3)
s3, d3 = c3[5], c3[-5]
drone = {"id": "scout-1", "structure": {"locomotion": {"rotors": 4}, "mass_kg": 2.5, "battery_wh": 60,
         "payload_bay_kg": 1.0, "sensors": ["camera", "lidar_20m"]}, "position": [*s3, 0]}
m3 = {"terrain": {"dims": 3, "seed": 7}, "robots": [drone],
      "items": [{"id": "radio", "weight_kg": 0.8, "pos": [*s3, 0]}],
      "goal": {"deliver": {"item": "radio", "to": [*d3, 0]}, "constraints": ["no_nofly", "battery_reserve"],
               "deadline_steps": 400}}
for alt in (2, 7):
    p3 = [{"robot": "scout-1", "skill": "pick", "args": {"item": "radio"}},
          {"robot": "scout-1", "skill": "fly_to", "args": {"pos": [*d3, 0], "alt": alt}},
          {"robot": "scout-1", "skill": "drop", "args": {"item": "radio"}}]
    t1 = tier1.check(m3, p3, {"reserve_frac": 0.2})
    sim3 = tier2.simulate(m3, p3); s = verifier.score(m3, sim3)
    print(f"DRONE alt={alt} tier1", t1); print(f"DRONE alt={alt} verify", s)
render(sim3, "out/drone.mp4", "drone alt7"); print("rendered out/drone.mp4", len(sim3.frames), "frames")
