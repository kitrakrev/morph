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
w("results", server.results("drone"))
lin = server.lineage("drone")
w("lineage", lin)
for h in lin:
    w(f"templates-{h['version']}", server.templates("drone", h["version"]))
w("trials-17", server.trials("drone", 17, None))
w("matrix", {"md": server.matrix_md()})
vids = [v for v in server.videos() if v.startswith("pair-")]
w("videos", vids)
w("feed", [])
for v in vids:
    shutil.copy(server.ROOT / "out" / v, dest / "out" / v)
shutil.copy(server.ROOT / "viewer" / "index.html", dest / "index.html")
print("exported", dest, len(lin), "versions,", len(vids), "videos")
