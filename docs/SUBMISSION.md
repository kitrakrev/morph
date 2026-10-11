# Morph — submission text

**Repo:** https://github.com/kitrakrev/morph · **Viewer:** https://kitrakrev.github.io/morph/

## Short description (paste into the Cerebral Valley form)

Morph adapts a rescue-robot fleet's planning software to new robot bodies on its own. You give it a robot's
structure (rotors, legs, wheels, battery, payload), a terrain and a goal. When a drone joins a ground fleet,
Morph's harness fails, finds out why from its own failure records in MongoDB Atlas, rewrites its own rules,
limits and workflow graph, and keeps a change only if a locked verifier shows more deliveries, no new safety
violations and no regression on the original missions. Same planner model throughout (Claude Haiku): held-out
drone missions went from **0/12 to 12/12 (n=12) with 0 violations**. On harder long-range missions it was never
tuned on, it scores 4/6 (n=6). The gate rejected every edit that added violations — e.g. v14, where letting the planner
read past episodes caused 30 no-fly violations. Built on MongoDB Atlas: versioned harness lineage,
`$rankFusion` over autoEmbedded (Voyage) failure summaries, a transaction on every promotion, time-series
metrics, checkpoints, and a change-stream live feed; Atlas Agent Engine memory (hosted server, service account) holding episodic, procedural, semantic and
taxonomic memory for each body. Raindrop receives an event
per trial and per gate decision; the evolution agent reads Raindrop's signals before each edit. A second body, a
fixed-wing VTOL, went from 6/12 to 12/12 held-out in one generation after Raindrop showed no-fly violations. It adapts the harness, not prompts or weights.

## Spoken opener (first 10 seconds)

"When a rescue team gets a new drone, someone spends weeks rewriting the software. Here's Morph's drone failing
every held-out mission — and here it is after the harness fixed itself: twelve out of twelve, same model."

## Claims checklist (what we can and can't say)

| Claim | Status |
|---|---|
| Held-out drone 0/12 → 12/12 (n=12), 0 violations, same planner model | verified (Atlas `trials`) |
| Hard tier: v17 held-out 4/6 (n=6) without tuning; 3 further edits all rejected | verified |
| Gate rejected v14 (30 no-fly violations) and v19 (min altitude 2, more violations) | verified |
| `$rankFusion` (vector + keyword) used by the evolution agent | verified (scoreDetails show both pipelines) |
| Transaction on promotion, change streams, time-series, checkpoints | verified in code + running |
| Agent Engine memory | live: hosted server; episodes from mission runs verified; save ~0.3–0.6 s, search ~0.3 s |
| Raindrop | 378 events + signals verified via Raindrop MCP; evolution agent reads signals |
| Transfer to new bodies | ground bodies need 0 generations (12/12 from bootstrap); cold-vs-warm **not measured** |
| Kill-and-resume | checkpoints written per node; **not demonstrated live yet** |
