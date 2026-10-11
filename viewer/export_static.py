"""Export a static snapshot of the viewer (JSON + videos) for GitHub Pages.

Usage: PYTHONPATH=. uv run python viewer/export_static.py <dest_dir>
"""
import json
import shutil
import sys
from pathlib import Path

from viewer import server

dest = Path(sys.argv[1])
(dest / "data").mkdir(parents=True, exist_ok=True)
(dest / "out").mkdir(exist_ok=True)
w = lambda name, obj: (dest / "data" / f"{name}.json").write_text(json.dumps(obj, default=str))
BODIES = ["drone", "fixed_wing_vtol"]
w("bodies", BODIES)
lin = []
for b in BODIES:  # drone keeps the un-suffixed files the page loads first
    sfx = "" if b == "drone" else f"-{b}"
    w("results" + sfx, server.results(b))
    lb = server.lineage(b)
    w("lineage" + sfx, lb)
    for h in lb:
        w(f"templates{sfx}-{h['version']}", server.templates(b, h["version"]))
    w("trials" + sfx, server.trials(b, None, None))
    lin += lb
w("matrix", {"md": server.matrix_md()})
vids = [v for v in server.videos() if v.startswith(("pair-", "evolution-", "attempts-"))]
w("videos", vids)
w("feed", [])
for v in vids:
    shutil.copy(server.ROOT / "out" / v, dest / "out" / v)
shutil.copy(server.ROOT / "viewer" / "index.html", dest / "index.html")
print("exported", dest, len(lin), "versions,", len(vids), "videos")
