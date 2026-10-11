"""Raindrop telemetry: one event per trial. No-op until the SDK is instrumented."""
from __future__ import annotations


def trial_event(trial: dict) -> None:
    return None
