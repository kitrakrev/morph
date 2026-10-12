"""Re-run a lineage from bootstrap with the full evidence path: 3-pipeline $rankFusion (text, keyword,
multimodal pictures) + Raindrop signals. Usage: PYTHONPATH=simple-backend uv run python -u
simple-backend/scripts/rerun_full_evidence.py <body> <new-lineage> [spec.yaml]"""
import json
import sys

import yaml

from engine.loop import adapt_curve

body, lineage = sys.argv[1], sys.argv[2]
spec = yaml.safe_load(open(sys.argv[3])) if len(sys.argv) > 3 else None
if spec:
    spec["id"] = lineage
r = adapt_curve(body, lineage, "cold", gens=4, target=1.0, spec=spec)
print("RESULT", json.dumps({k: r[k] for k in ("lineage", "gens_to_adapt", "curve")}))
