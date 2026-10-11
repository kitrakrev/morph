"""Read-only evidence viewer for Morph. Reads Atlas, serves one HTML page with a sidebar.

Usage: PYTHONPATH=. uv run python viewer/server.py   ->  http://localhost:8765
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict, deque
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from engine.store import db

ROOT = Path(__file__).resolve().parent.parent
PORT = int(os.getenv("PORT", "8765"))


def _era_cutoff() -> float:
    """Trials before the first Claude-planner evaluation belong to the earlier run (free model, physics v1)."""
    ev = db().evaluations.find_one({"model": {"$regex": "^claude-code"}}, sort=[("ts", 1)])
    return (ev["ts"] - 3600) if ev else 0.0


def results(body: str) -> list[dict]:
    cutoff = _era_cutoff()
    latest: dict = {}
    for t in db().trials.find({"body": body}, {"harness_version": 1, "split": 1, "mission_id": 1, "score": 1, "ts": 1}):
        k = (t["harness_version"], t["split"], t["mission_id"])
        if k not in latest or t["ts"] > latest[k]["ts"]:
            latest[k] = t
    agg = defaultdict(lambda: defaultdict(lambda: {"ok": 0, "n": 0, "viol": 0, "ts": 0}))
    for (v, split, _), t in latest.items():
        a = agg[v][split]
        a["ok"] += int(t["score"]["success"])
        a["n"] += 1
        a["viol"] += t["score"]["violations_total"]
        a["ts"] = max(a["ts"], t["ts"])
    hs = {h["version"]: h for h in db().harnesses.find({"body_type": body}, {"version": 1, "verdict": 1, "diff": 1})}
    evs = {e["version"]: e for e in db().evaluations.find({"body": body})}
    rows = []
    for v in sorted(set(agg) | set(hs)):
        sp = agg.get(v, {})
        latest_ts = max([x["ts"] for x in sp.values()] or [0])
        diff = hs.get(v, {}).get("diff")
        fields = [d["field"] for d in diff] if isinstance(diff, list) else ([diff["field"]] if diff else [])
        rows.append({"version": v, "verdict": hs.get(v, {}).get("verdict", "?"), "fields": fields,
                     "era": "current" if latest_ts >= cutoff else "earlier",
                     "splits": {s: {k: x[k] for k in ("ok", "n", "viol")} for s, x in sp.items()},
                     "reason": evs.get(v, {}).get("reason"), "prediction": evs.get(v, {}).get("prediction"),
                     "measured": evs.get(v, {}).get("measured")})
    return rows


def lineage(body: str) -> list[dict]:
    out = []
    evs = {e["version"]: e for e in db().evaluations.find({"body": body})}
    for h in db().harnesses.find({"body_type": body}).sort("version", 1):
        e = evs.get(h["version"], {})
        out.append({"version": h["version"], "parent": h.get("parent"), "verdict": h.get("verdict"),
                    "diff": h.get("diff"), "rationale": h.get("rationale"), "prediction": h.get("prediction"),
                    "reason": e.get("reason"), "checks": e.get("checks"), "metrics": e.get("metrics"),
                    "model": e.get("model"), "retrieved": e.get("retrieved_failures", [])[:5],
                    "workflow": h.get("workflow"), "rules": h.get("rules"), "budgets": h.get("budgets"),
                    "transferred_from": h.get("transferred_from")})
    return out


def trials(body: str, version: int | None, split: str | None) -> list[dict]:
    q = {"body": body}
    if version is not None:
        q["harness_version"] = version
    if split:
        q["split"] = split
    return [{"mission": t["mission_id"], "version": t["harness_version"], "split": t["split"],
             "ok": t["score"]["success"], "energy": t["score"]["energy_wh"], "viol": t["score"]["violations_total"],
             "template": t.get("which"), "diagnosis": t.get("diagnosis", ""),
             "actions": " > ".join(c["skill"] for c in t.get("program", []))}
            for t in db().trials.find(q).sort("ts", -1).limit(200)]


FEED: deque = deque(maxlen=60)


def watch_changes():
    """Atlas change stream on the morph DB: live feed of trials, harness versions and gate decisions."""
    pipeline = [{"$match": {"operationType": {"$in": ["insert", "replace"]},
                            "ns.coll": {"$in": ["trials", "harnesses", "evaluations"]}}}]
    while True:
        try:
            with db().watch(pipeline, full_document="updateLookup") as stream:
                for ch in stream:
                    d, coll = ch.get("fullDocument") or {}, ch["ns"]["coll"]
                    if coll == "trials":
                        msg = (f"trial {d.get('body')} v{d.get('harness_version')} {d.get('mission_id')}: "
                               f"{'DELIVERED' if d.get('score', {}).get('success') else 'FAILED ' + d.get('diagnosis', '')[:70]}")
                    elif coll == "evaluations":
                        msg = (f"GATE {d.get('body')} v{d.get('version')}: {'ACCEPTED' if d.get('accepted') else 'REJECTED'}"
                               f" ({d.get('reason')})")
                    else:
                        msg = f"harness {d.get('body_type')} v{d.get('version')} -> {d.get('verdict')}"
                    FEED.appendleft({"ts": time.strftime("%H:%M:%S"), "coll": coll, "msg": msg})
        except Exception as e:
            FEED.appendleft({"ts": time.strftime("%H:%M:%S"), "coll": "error", "msg": f"change stream: {e}"})
            time.sleep(5)


def templates(body: str, version: int) -> dict:
    """One example program per template kind (primary / fallback) actually run for this version."""
    out = {}
    for which in ("primary", "fallback"):
        t = db().trials.find_one({"body": body, "harness_version": version, "which": which}, sort=[("ts", -1)])
        if t:
            out[which] = [f"{c['skill']}({', '.join(f'{k}={v}' for k, v in c['args'].items() if k != 'item')})"
                          for c in t["program"]]
    return out


def videos() -> list[str]:
    return sorted(p.name for p in (ROOT / "out").glob("*.mp4"))


def matrix_md() -> str:
    p = ROOT / "docs" / "matrix.md"
    return p.read_text() if p.exists() else ""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        body = q.get("body", "drone")
        try:
            if u.path == "/api/results":
                return self._json(results(body))
            if u.path == "/api/lineage":
                return self._json(lineage(body))
            if u.path == "/api/trials":
                return self._json(trials(body, int(q["version"]) if "version" in q else None, q.get("split")))
            if u.path == "/api/videos":
                return self._json(videos())
            if u.path == "/api/templates":
                return self._json(templates(body, int(q.get("version", 1))))
            if u.path == "/api/feed":
                return self._json(list(FEED))
            if u.path == "/api/matrix":
                return self._json({"md": matrix_md()})
        except Exception as e:  # surface errors to the page instead of a blank screen
            return self._json({"error": str(e)})
        if u.path.startswith("/out/"):
            self.path = u.path
            return super().do_GET()
        self.path = "/viewer/index.html"
        return super().do_GET()

    def translate_path(self, path):
        return str(ROOT / path.lstrip("/"))


if __name__ == "__main__":
    threading.Thread(target=watch_changes, daemon=True).start()
    print(f"Morph viewer on http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
