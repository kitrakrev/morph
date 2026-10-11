# Morph — the fleet that learns new bodies

**Plug in a robot. The harness does the rest.**

Robot fleets are mixed and keep changing: bodies get damaged, refitted with new payloads, or replaced by
another vendor's model. Today every change means engineers rewrite rules, limits and routing by hand, then
re-test — weeks per body. Morph is a self-evolving **harness** around a fixed planner model. It reads a new
body's structure, guesses what it can do, learns from its own failures, and keeps only changes that are
proven on missions it never saw. Every lesson is stored in MongoDB Atlas and can transfer to the next body.

**The model never changes. The harness does.**

Built solo at The Harness Engineering & Model Wrangling #SFTechWeek Hackathon (Oct 11 2026).
Problem statement: **1 — Recursive Harnessing** (primary); touches **2 — Long Horizon** through
persistent Atlas memory and checkpoint/resume.

---

## How it works

```
 user supplies: robot structure + terrain + goal          (never rules, tools or code)
        │
        ▼
 bootstrap harness v1 from structure ──► harnesses (Atlas, versioned, parent links)
        │
        ▼   Loop 1: missions                         Loop 2: evolution (on plateau)
 planner (fixed free model) writes a plan template    evolution agent reads failures
        │                                             ($rankFusion: autoEmbed vector + keyword)
 workflow graph from the harness:                     proposes ONE bounded edit + prediction
   plan → tier1_check → replan → tier2_sim → verify          │
        │                                             candidate evaluated on train + regression
 locked verifier scores against TRUE physics                 │
        │                                             gate: legal, train ↑, safety not worse,
 trials, failures, episodic memory → Atlas            no regression, cost per success
 one event per trial + failure signals → Raindrop     verdict + evaluation in one transaction
```

**Fixed code (no agent edits it):** `sim/` world, physics, skills, tier 1/2, verifier, renderer;
`engine/runner.py`, `engine/gate.py`.

**Evolvable data (the harness document):** workflow graph, rules, constraints, tool specs, beliefs
(energy model, reserve), reports read, retrieval policy, model routing, budgets, composed skills.
Locked: `constraints.nofly` and `constraints.collision` stay `hard`; tools must be primitives.

## Results so far (drone, same planner model throughout)

| Version | Edit | Train | Regression | Violations | Gate |
|---|---|---|---|---|---|
| v1 | bootstrap from structure | 1/6 | 0/4 | 3 | — |
| v5 | +tier1_check, +replan, energy belief 0.8 | 2/6 | 1/4 | 2 | rejected: cost (old step-count rule) |
| **v8** | +tier1_check, +replan, energy belief 0.75, reserve 10% | **2/6** | **1/4** | **2** | **accepted** |
| v14 | planner reads 2 past episodes | 2/6 | 1/4 | **30** | rejected: safety worse |

13 edits proposed, 1 accepted, 12 rejected. Prediction vs measured is stored per edit in `evaluations`.
The search space is real: a hand-checked best plan inside the harness's own options solves **12/12**
held-out drone missions; a naive plan solves 2/12. The current plateau is a planner rule (charge *after*
pickup) the free evolution model has not found yet.

Held-out missions are **not run yet** for any promoted version. Full matrix status: [docs/matrix.md](docs/matrix.md).

## Evaluation matrix

| Axis | Values |
|---|---|
| Bodies | rover, tracked, quadruped, drone, rail cart, **hexapod (held-out body)** |
| Dimensions | 1D corridor (rail), 2D city (ground), 3D voxel city (drone) |
| Terrain difficulty | easy (open, many chargers), medium (rubble), hard (more rubble, 2 chargers, 1 river ford, big no-fly) |
| Mission types | single delivery, heavy payload, deadline-tight, multi-stop, relay (two bodies, handoff) |

90 cells × (train 6 / regression 4 / held-out 12) missions, generated from seeds in 3.5 s, each checked
solvable under the true physics. Splits use disjoint seed ranges; held-out outcomes never reach
agent-readable memory. Rail cart multi-stop and relay cells are short (a single track has too few distinct points).

Target numbers for the pitch: held-out success per body; **generations to adapt per body, falling as
lessons transfer**; safety violations on promoted versions.

## MongoDB Atlas features used

| Feature | Job in Morph |
|---|---|
| Versioned documents, unique `(body_type, version)` index | harness lineage v1 → v2 → … with parent links and diffs |
| **Multi-document transaction** | harness verdict + evaluation record written atomically |
| **Automated embeddings (autoEmbed, voyage-4-lite)** | failure summaries and four memory types embedded by Atlas on insert |
| **Vector Search + Atlas Search + `$rankFusion`** | hybrid retrieval of similar past failures for the evolution agent |
| Time-series collection | per-trial metrics for curves |
| Checkpoints collection | runner state per workflow node; a killed run resumes |
| Memory collections (semantic / episodic / procedural / taxonomic) | Agent Engine–shaped memory; user_id = body category |
| LLM response cache (unique key) | deterministic re-runs within the free-model budget |

Atlas Agent Engine: CLI signed in and a memory-only project scaffolded; the memory server needs a Voyage
key and was not provisioned today, so memory runs on plain collections with the same API shape.

## Built today vs used

**Built today:** everything in `sim/`, `engine/`, `scripts/` — the N-D world, physics, skills, tiers,
verifier, renderer, mission matrix, harness schema, planner, runner, evolution agent, gate, memory,
telemetry, reports.

**Used:** MongoDB Atlas (sandbox cluster), OpenRouter free models (`nvidia/nemotron-3-super-120b-a12b:free`,
`nvidia/nemotron-3-ultra-550b-a55b:free`), Raindrop (Python SDK), numpy, matplotlib, ffmpeg, pymongo.
Not used: MuJoCo, real robots, any training or fine-tuning.

## Run it

```bash
uv sync
cp .env.example .env   # MONGODB_URI, OPENROUTER_API_KEY, RAINDROP_WRITE_KEY
PYTHONPATH=. uv run python -c "from engine.store import ensure_schema; ensure_schema()"
PYTHONPATH=. uv run python -m engine.loop drone --gens 4        # evolve
PYTHONPATH=. uv run python scripts/replay.py drone train-drone-105 1 8   # MP4 replays in out/
PYTHONPATH=. uv run python scripts/matrix_report.py              # matrix status
```

## Honesty notes

- Physics is simulated and simple (grid cells, Wh per cell, quadratic flight payload cost).
- Held-out sets are small (12 missions per cell); results above are train/regression only so far.
- Free models cap at 50 requests/day; one plan template per harness version keeps evaluation cheap and
  deterministic, at the cost of per-mission LLM reasoning.
- The evolution agent sees train/regression failures only, never held-out missions or the verifier.

## Prior art and how Morph differs

Self-improving harness work (GEPA, Self-Harness, Darwin Gödel Machine, RRSI) evolves prompts or agent code
on fixed tasks; RRSI shows unregularized harness evolution often fails to transfer. Morph keeps the model
fixed, evolves only harness data, gates on regression with a locked verifier, and targets a different
shift: **a new robot body**, measured by how fast the harness adapts.

## Real robots each body stands in for

rover — Clearpath Husky/Jackal · tracked — FLIR PackBot, Telerob tEODor · quadruped — Boston Dynamics Spot,
ANYbotics ANYmal, Unitree B2 · drone — DJI Matrice 350, Skydio X10 · rail cart — rail-guided inspection
robots in tunnels, mines, data centers · hexapod — research platforms.

Next: multi-floor buildings (stairs as 3D for legged bodies), relay planning, quadruped/hexapod transfer runs.
