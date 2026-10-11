"""Replay renderer: 1D strip / 2D top-down / 3D isometric -> MP4. Fixed code."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from sim.skills import Sim  # noqa: E402
from sim.world import T  # noqa: E402

COLORS = {T.FLOOR: "#e8e4da", T.RUBBLE: "#9c7c5a", T.STAIRS: "#c9a96e", T.WATER: "#6fa8dc",
          T.WALL: "#444444", T.AIR: "#ffffff", T.NOFLY: "#e06666", T.CHARGER: "#6aa84f", T.RAIL: "#8e7cc3"}
CMAP = ListedColormap([COLORS[T(i)] for i in range(len(T))])
ROBOT_COLORS = ["#cc0000", "#0b5394", "#bf9000", "#38761d", "#741b47"]


def _ground_image(sim: Sim) -> np.ndarray:
    g = sim.world.grid
    if g.ndim == 1:
        return g[np.newaxis, :]
    if g.ndim == 2:
        return g.T
    top = g[:, :, 0].copy()
    top[(g[:, :, 1:] == T.WALL).any(axis=2)] = T.WALL
    top[(g[:, :, 1:] == T.NOFLY).any(axis=2) & (top != T.WALL)] = T.NOFLY
    return top.T


def render(sim: Sim, path: str, title: str = "", fps: int = 20, max_frames: int = 400) -> str:
    frames = sim.frames
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    fig, ax = plt.subplots(figsize=(6, 6 if sim.world.dims > 1 else 2))
    ax.imshow(_ground_image(sim), cmap=CMAP, vmin=0, vmax=len(T) - 1, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    rids = list(sim.robots)
    dots = {rid: ax.plot([], [], "o", ms=9, mec="black", color=ROBOT_COLORS[i % 5], label=rid)[0]
            for i, rid in enumerate(rids)}
    trails = {rid: ax.plot([], [], "-", lw=1.5, color=ROBOT_COLORS[i % 5], alpha=0.7)[0] for i, rid in enumerate(rids)}
    hud = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace",
                  bbox={"facecolor": "white", "alpha": 0.8})
    ax.legend(loc="lower right", fontsize=7)
    ax.set_title(title, fontsize=9)
    hist = {rid: ([], []) for rid in rids}

    def xy(p):
        return (p[0], 0) if len(p) == 1 else (p[0], p[1])

    def update(k):
        f = frames[k]
        lines = [f"t={f['t']}"]
        for rid, st in f["robots"].items():
            x, y = xy(st["pos"])
            hist[rid][0].append(x)
            hist[rid][1].append(y)
            dots[rid].set_data([x], [y])
            dots[rid].set_marker("o" if st["alive"] else "X")
            trails[rid].set_data(*hist[rid])
            alt = f" z={st['pos'][2]}" if len(st["pos"]) == 3 else ""
            lines.append(f"{rid}: {st['battery']:.1f}Wh{alt}{' DEAD' if not st['alive'] else ''}")
        hud.set_text("\n".join(lines))
        return [*dots.values(), *trails.values(), hud]

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)
    return path
