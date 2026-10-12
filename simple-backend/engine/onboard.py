"""Onboard a new robot from its spec file: bootstrap from structure + declared capabilities, score
held-out missions on its terrain, evolve until it passes, report.

Usage: PYTHONPATH=. uv run python -m engine.onboard robots/medic-drone.yaml [--gens 3]
"""
from __future__ import annotations

import argparse
import json

import yaml

from engine.harness import UnsupportedBody, bootstrap
from engine.loop import adapt_curve
from sim.physics import Body

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--gens", type=int, default=3)
    ap.add_argument("--lineage", help="lineage name in Atlas (default: the spec's id)")
    a = ap.parse_args()
    spec = yaml.safe_load(open(a.spec))
    if a.lineage:
        spec["id"] = a.lineage
    try:
        bootstrap(spec["structure"], spec.get("capabilities"))  # validates locomotion vocabulary first
    except UnsupportedBody as e:
        raise SystemExit(str(e))
    kind = Body.from_structure(spec["structure"]).kind
    print(f"[onboard] {spec['id']}: body={kind}, capabilities={sorted(spec.get('capabilities') or ['derived'])}, "
          f"terrain={spec.get('terrain')}")
    r = adapt_curve(kind, spec["id"], "cold", gens=a.gens, target=1.0, spec=spec)
    print(json.dumps({"robot": spec["id"], "gens_to_adapt": r["gens_to_adapt"],
                      "curve": [(c["version"], f"{c['heldout']}/{c['n']}", c["viol"]) for c in r["curve"]]}))
