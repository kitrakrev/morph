"""Replay renderer: top-down map (+ altitude strip in 3D), legend, battery HUD -> MP4. Fixed code.

`render` draws one run; `render_pair` draws two runs of the same mission side by side,
synced on simulation time t.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from sim.skills import Sim  # noqa: E402
from sim.world import T  # noqa: E402

COLORS = {T.FLOOR: "#ece8de", T.RUBBLE: "#9c7c5a", T.STAIRS: "#c9a96e", T.WATER: "#6fa8dc",
          T.WALL: "#555555", T.AIR: "#ffffff", T.NOFLY: "#e06666", T.CHARGER: "#4caf50", T.RAIL: "#8e7cc3"}
CMAP = ListedColormap([COLORS[T(i)] for i in range(len(T))])
ROBOT_COLORS = ["#d32f2f", "#1565c0", "#f9a825", "#2e7d32", "#6a1b9a"]
GREEN, RED = "#2e7d32", "#c62828"
FPS, HOLD_S, MAX_FRAMES = 6, 2.0, 90


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


def _xy(p):
    return (p[0], 0) if len(p) == 1 else (p[0], p[1])


def _legend(fig):
    handles = [Patch(color=COLORS[T.WALL], label="building"), Patch(color=COLORS[T.RUBBLE], label="rubble"),
               Patch(color=COLORS[T.WATER], label="water"), Patch(color=COLORS[T.NOFLY], label="no-fly (z<6)"),
               Patch(color=COLORS[T.CHARGER], label="charger"), Patch(color=COLORS[T.RAIL], label="rail"),
               Line2D([], [], marker="s", ls="", color="#ff9800", mec="k", label="pickup"),
               Line2D([], [], marker="*", ls="", color="#ffeb3b", mec="k", ms=11, label="target"),
               Line2D([], [], marker="X", ls="", color="k", label="battery death")]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=7, frameon=False)


class Panel:
    """One run drawn into a column of axes: map (+ altitude strip for 3D)."""

    def __init__(self, fig, gs_col, sim: Sim, mission: dict | None, title: str, result: str, ok: bool):
        self.sim, self.frames = sim, sim.frames
        three_d = sim.world.dims == 3
        if three_d:
            sub = gs_col.subgridspec(2, 1, height_ratios=[4, 1], hspace=0.25)
            self.ax, self.alt_ax = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        else:
            self.ax, self.alt_ax = fig.add_subplot(gs_col), None
        ax = self.ax
        ax.imshow(_ground_image(sim), cmap=CMAP, vmin=0, vmax=len(T) - 1, origin="lower", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9, pad=16)
        ax.text(0.5, 1.01, result, transform=ax.transAxes, ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=GREEN if ok else RED)
        if mission:
            for it in mission.get("items", []):
                ax.plot(*_xy(it["pos"]), "s", color="#ff9800", mec="k", ms=8, zorder=4)
                ax.annotate("pickup", _xy(it["pos"]), xytext=(4, 4), textcoords="offset points", fontsize=6)
            goals = mission["goal"].get("deliveries") or [mission["goal"]["deliver"]]
            for g in goals:
                ax.plot(*_xy(g["to"]), "*", color="#ffeb3b", mec="k", ms=14, zorder=4)
                ax.annotate("target", _xy(g["to"]), xytext=(4, -9), textcoords="offset points", fontsize=6)
        self.rids = list(sim.robots)
        self.dot = {r: ax.plot([], [], "o", ms=9, mec="k", color=ROBOT_COLORS[i % 5], zorder=5)[0]
                    for i, r in enumerate(self.rids)}
        self.trail = {r: ax.plot([], [], "-", lw=1.6, color=ROBOT_COLORS[i % 5], alpha=0.8, zorder=3)[0]
                      for i, r in enumerate(self.rids)}
        self.hud = ax.text(0.01, 0.99, "", transform=ax.transAxes, va="top", fontsize=7, family="monospace",
                           bbox={"facecolor": "white", "alpha": 0.85, "lw": 0})
        # battery bar (axes coords, top-right) with reserve line
        self.bat_bg = Rectangle((0.70, 0.93), 0.28, 0.04, transform=ax.transAxes, fc="white", ec="k", lw=0.8, zorder=6)
        self.bat = Rectangle((0.70, 0.93), 0.28, 0.04, transform=ax.transAxes, fc=GREEN, zorder=7)
        ax.add_patch(self.bat_bg)
        ax.add_patch(self.bat)
        reserve = (mission or {}).get("goal", {}).get("reserve_frac", 0.2)
        self.reserve = reserve
        ax.plot([0.70 + 0.28 * reserve] * 2, [0.92, 0.98], transform=ax.transAxes, color="k", lw=1.2, zorder=8)
        ax.text(0.70 + 0.28 * reserve, 0.985, "reserve", transform=ax.transAxes, fontsize=5.5, ha="center", zorder=8)
        if self.alt_ax is not None:
            a = self.alt_ax
            a.set_xlim(0, max(f["t"] for f in self.frames) + 1)
            a.set_ylim(0, sim.world.size[2])
            a.axhspan(1, 6, color=COLORS[T.NOFLY], alpha=0.15)
            a.text(1, 5.2, "no-fly band z<6 (where zone exists)", fontsize=5.5, color=RED)
            a.set_ylabel("altitude", fontsize=7)
            a.set_xlabel("t (steps)", fontsize=7)
            a.tick_params(labelsize=6)
            self.alt_line = {r: a.plot([], [], "-", color=ROBOT_COLORS[i % 5])[0] for i, r in enumerate(self.rids)}
        self.hist = {r: ([], [], [], []) for r in self.rids}
        self.k = 0

    def frame_at(self, t):
        while self.k + 1 < len(self.frames) and self.frames[self.k + 1]["t"] <= t:
            self.k += 1
        return self.frames[self.k]

    def draw(self, t):
        f = self.frame_at(t)
        lines = [f"t={f['t']}"]
        for r, st in f["robots"].items():
            x, y = _xy(st["pos"])
            h = self.hist[r]
            if not h[0] or (h[0][-1], h[1][-1]) != (x, y):
                h[0].append(x)
                h[1].append(y)
                h[2].append(f["t"])
                h[3].append(st["pos"][2] if len(st["pos"]) == 3 else 0)
            self.dot[r].set_data([x], [y])
            self.dot[r].set_marker("o" if st["alive"] else "X")
            self.dot[r].set_color(ROBOT_COLORS[self.rids.index(r) % 5] if st["alive"] else "k")
            self.trail[r].set_data(h[0], h[1])
            if self.alt_ax is not None:
                self.alt_line[r].set_data(h[2], h[3])
            alt = f" z={st['pos'][2]}" if len(st["pos"]) == 3 else ""
            load = f" carrying {','.join(st['carrying'])}" if st["carrying"] else ""
            lines.append(f"{r}{alt}{load}{'  DEAD' if not st['alive'] else ''}")
        r0 = self.rids[0]
        frac = f["robots"][r0]["battery"] / self.sim.robots[r0].body.battery_wh
        self.bat.set_width(0.28 * max(0.0, frac))
        self.bat.set_facecolor(GREEN if frac >= self.reserve else RED)
        lines.append(f"battery {f['robots'][r0]['battery']:.1f}Wh ({frac:.0%})")
        self.hud.set_text("\n".join(lines))


def _timeline(panels) -> list[int]:
    ts = sorted({f["t"] for p in panels for f in p.frames})
    if len(ts) > MAX_FRAMES:
        ts = [ts[i] for i in np.linspace(0, len(ts) - 1, MAX_FRAMES).astype(int)]
    return ts + [ts[-1]] * int(FPS * HOLD_S)


def _save(fig, panels, path):
    ts = _timeline(panels)

    def update(i):
        for p in panels:
            p.draw(ts[i])
        return []

    anim = animation.FuncAnimation(fig, update, frames=len(ts), blit=False)
    anim.save(path, writer=animation.FFMpegWriter(fps=FPS, bitrate=2400), dpi=130)
    plt.close(fig)
    return path


def render(sim: Sim, path: str, title: str = "", result: str = "", ok: bool = True, mission: dict | None = None) -> str:
    fig = plt.figure(figsize=(6, 7.2 if sim.world.dims == 3 else 6.6))
    gs = fig.add_gridspec(1, 1, left=0.04, right=0.96, top=0.9, bottom=0.1)
    p = Panel(fig, gs[0], sim, mission, title, result, ok)
    _legend(fig)
    return _save(fig, [p], path)


def render_pair(left: tuple, right: tuple, path: str, suptitle: str = "") -> str:
    """left/right = (sim, mission, title, result, ok). Synced on simulation time."""
    three_d = left[0].world.dims == 3
    fig = plt.figure(figsize=(12, 7.4 if three_d else 6.8))
    gs = fig.add_gridspec(1, 2, left=0.05, right=0.97, top=0.88, bottom=0.11, wspace=0.12)
    panels = [Panel(fig, gs[i], *args) for i, args in enumerate((left, right))]
    tmax = max(f["t"] for p in panels for f in p.frames) + 1
    for p in panels:
        if p.alt_ax is not None:
            p.alt_ax.set_xlim(0, tmax)  # shared time axis so the two runs line up
    _legend(fig)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    return _save(fig, panels, path)
