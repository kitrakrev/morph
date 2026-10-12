#!/usr/bin/env bash
# As each full-evidence rerun finishes: render its per-attempt video + why-card, re-export the site, push.
cd "$(dirname "$0")/../.."
pick() { PYTHONPATH=simple-backend uv run python - "$1" <<'PY' 2>/dev/null | grep -v DEBUG | tail -1
import sys
from engine.store import db
lin = sys.argv[1]
acc = [h["version"] for h in db().harnesses.find({"body_type": lin, "verdict": "accepted"})]
best = None
for t in db().trials.find({"body": lin, "harness_version": 1, "split": {"$in": ["train", "heldout"]}, "score.success": False}):
    n = len(db().trials.distinct("harness_version", {"body": lin, "mission_id": t["mission_id"]}))
    won = any(db().trials.find_one({"body": lin, "harness_version": v, "mission_id": t["mission_id"], "score.success": True}) for v in acc)
    score = n + (100 if won else 0)
    if not best or score > best[1]:
        best = (t["mission_id"], score)
print(best[0] if best else "")
PY
}
publish() {
  lin=$1; mid=$(pick $lin)
  echo "[$(date +%H:%M)] [$lin] $(grep '^RESULT' out/rerun-$lin.log | cut -c1-160) -> render $mid"
  [ -n "$mid" ] && PYTHONPATH=simple-backend uv run python simple-backend/scripts/attempts_with_params.py $lin "$mid" 2>&1 | grep -v DEBUG | tail -1 \
    && cp out/attempts-with-params-$lin.mp4 docs/out/
  PYTHONPATH=simple-backend uv run python simple-backend/viewer/export_static.py docs 2>&1 | grep -v DEBUG | tail -1
  git add -A docs && git commit -qm "data: $lin full-evidence rerun results + video

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>" && git pull --rebase -q && git push -q && echo "[$lin] pushed $(git log --oneline -1)"
}
pending="drone-r2 fixed_wing_vtol-r2 medic-drone-r4"
while [ -n "$pending" ]; do            # check every minute; publish each run the moment it finishes
  next=""
  for lin in $pending; do
    if grep -q '^RESULT' out/rerun-$lin.log 2>/dev/null; then publish $lin; else next="$next $lin"; fi
  done
  pending=$(echo $next)
  [ -n "$pending" ] && sleep 60
done
echo "[$(date +%H:%M)] all reruns published"
