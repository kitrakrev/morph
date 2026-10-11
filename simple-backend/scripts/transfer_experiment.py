"""Quadruped learns first; then the held-out hexapod starts cold vs warm (taxonomic transfer).

Usage: PYTHONPATH=. uv run python -u scripts/transfer_experiment.py
"""
import json

from engine import transfer
from engine.loop import adapt_curve
from engine.store import db
from sim.missions import BODIES

results = [adapt_curve("quadruped", "quadruped", "cold", gens=3)]
print("lesson stored:", transfer.record_lessons("quadruped", BODIES["quadruped"]))
results.append(adapt_curve("hexapod", "hexapod-cold", "cold", gens=3))
results.append(adapt_curve("hexapod", "hexapod-warm", "warm", gens=3))
db().experiments.replace_one({"name": "transfer"}, {"name": "transfer", "results": results}, upsert=True)
print(json.dumps([{k: r[k] for k in ("lineage", "start", "gens_to_adapt")} | {"curve": [(c["heldout"], c["n"]) for c in r["curve"]]}
                  for r in results], indent=1))
