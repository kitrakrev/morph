"""Raindrop telemetry.

- `mission_trial` event per trial: user_id = body type, convo_id = harness version.
  Failure kinds are attached as signals so Raindrop can cluster them.
- `harness_llm` event per real LLM call (planner template, evolution proposal).
Held-out trials are tagged split=heldout; the evolution agent never reads them.
"""
from __future__ import annotations

import atexit
import json
import os

import raindrop.analytics as raindrop
from dotenv import load_dotenv

load_dotenv()
_ON = bool(os.getenv("RAINDROP_WRITE_KEY"))
if _ON:
    raindrop.init(os.getenv("RAINDROP_WRITE_KEY"))
    raindrop.set_debug_logs(os.getenv("RAINDROP_DEBUG", "1") == "1")  # turn off once events are confirmed
    atexit.register(raindrop.shutdown)


def trial_event(trial: dict, model: str | None = None) -> str | None:
    if not _ON:
        return None
    s = trial["score"]
    event_id = trial["run_id"]
    seq = " > ".join(c["skill"] for c in trial["program"])
    raindrop.track_ai(
        user_id=trial["body"],
        event="mission_trial",
        event_id=event_id,
        model=model,
        convo_id=f"{trial['body']}-v{trial['harness_version']}",
        input=json.dumps({"mission": trial["mission_id"], "program": trial["program"]}),
        output=("DELIVERED safely" if s["success"] else f"FAILED: {trial['diagnosis']}") + f" | {seq}",
        properties={"harness_version": trial["harness_version"], "split": trial["split"],
                    "success": s["success"], "delivered": s["delivered"], "energy_wh": s["energy_wh"],
                    "time": s["time"], "violations_total": s["violations_total"],
                    "template": trial.get("which"), **{f"v_{k}": v for k, v in s["safety_violations"].items()}},
    )
    kinds = [k for k, v in s["safety_violations"].items() if v]
    if s["reserve_breach"]:
        kinds.append("reserve_breach")
    for k in kinds:
        raindrop.track_signal(event_id=event_id, name=k if k != "nofly" else "nofly_violation",
                              signal_type="default", sentiment="NEGATIVE")
    if s["success"]:
        raindrop.track_signal(event_id=event_id, name="delivered", signal_type="default", sentiment="POSITIVE")
    return event_id


def llm_event(role: str, model: str, system: str, user: str, output: dict, body: str = "fleet",
              version: int | None = None) -> None:
    if not _ON:
        return
    raindrop.track_ai(
        user_id=body, event=f"harness_{role}", model=model,
        convo_id=f"{body}-v{version}" if version is not None else None,
        input=user[:20000], output=json.dumps(output)[:20000],
        properties={"role": role, "system_prompt": system[:4000]},
    )


def flush() -> None:
    if _ON:
        raindrop.flush()
