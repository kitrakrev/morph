"""OpenRouter client: free models, Atlas response cache, daily-budget guard.

Free models share a 50 requests/day cap on this key, so every response is
cached by prompt hash and the guard refuses calls once the budget is spent.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

import requests
from dotenv import load_dotenv

from engine.store import db

load_dotenv()
URL = "https://openrouter.ai/api/v1"
HEADERS = {"HTTP-Referer": "https://github.com/kitrakrev/morph", "X-Title": "Morph"}
MODELS = {
    "cheap": ["nvidia/nemotron-3-super-120b-a12b:free", "nvidia/nemotron-3-ultra-550b-a55b:free"],
    "strong": ["nvidia/nemotron-3-ultra-550b-a55b:free", "nvidia/nemotron-3-super-120b-a12b:free"],
}
FREE_BUDGET_STOP = int(os.getenv("MORPH_FREE_BUDGET_STOP", "45"))


class BudgetExhausted(RuntimeError):
    pass


def _key() -> str:
    return os.environ["OPENROUTER_API_KEY"]


def free_usage() -> dict:
    r = requests.get(f"{URL}/key", headers={"Authorization": f"Bearer {_key()}"}, timeout=15).json()["data"]
    return r.get("free_model_daily_requests", {})


def _extract_json(text: str):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=0)
    obj, _ = json.JSONDecoder().raw_decode(text[start:])  # ignore any trailing text
    return obj


CLAUDE_MODEL = os.getenv("MORPH_CLAUDE_MODEL", "sonnet")


def claude_json(system: str, user: str, timeout: int = 180, model: str | None = None, retries: int = 1,
                tools: list[str] | None = None, cache: bool = True) -> dict:
    """Claude via Claude Code headless mode (`claude -p`). Cached in Atlas by prompt hash."""
    import subprocess
    model = model or CLAUDE_MODEL
    key = hashlib.sha256(json.dumps(["claude", model, system, user]).encode()).hexdigest()
    hit = db().llm_cache.find_one({"key": key}) if cache else None
    if hit:
        return {**hit["response"], "_cached": True}
    t0, last = time.time(), None
    extra = ["--allowedTools", ",".join(tools)] if tools else []
    for _ in range(retries + 1):
        proc = subprocess.run(["claude", "-p", f"{system}\n\n{user}", "--model", model,
                               "--output-format", "json", *extra], capture_output=True, text=True, timeout=timeout,
                              cwd="/tmp")
        try:
            msgs = json.loads(proc.stdout)
            res = next(m for m in reversed(msgs if isinstance(msgs, list) else [msgs]) if m.get("type") == "result")
            if res.get("is_error"):
                raise RuntimeError(f"claude -p error: {res.get('result')}")
            parsed = _extract_json(res["result"])
            break
        except (ValueError, StopIteration, RuntimeError) as e:
            last = e
    else:
        raise RuntimeError(f"claude -p failed: {last}")
    out = {"json": parsed, "model": f"claude-code:{model}",
           "latency_s": round(time.time() - t0, 2), "cost_usd": res.get("total_cost_usd")}
    if cache:
        db().llm_cache.insert_one({"key": key, "tier": "claude", "response": out, "ts": time.time()})
    return out


def chat_json(system: str, user: str, tier: str = "cheap", max_tokens: int = 1500) -> dict:
    """Return parsed JSON from the first model that answers. Cached in Atlas forever."""
    key = hashlib.sha256(json.dumps([tier, system, user]).encode()).hexdigest()
    hit = db().llm_cache.find_one({"key": key})
    if hit:
        return {**hit["response"], "_cached": True}
    usage = free_usage()
    if usage.get("used", 0) >= FREE_BUDGET_STOP:
        raise BudgetExhausted(f"free requests used {usage.get('used')}/{usage.get('limit')}")
    last_err = None
    for model in MODELS[tier]:
        t0 = time.time()
        try:
            r = requests.post(f"{URL}/chat/completions", timeout=120,
                              headers={"Authorization": f"Bearer {_key()}", **HEADERS},
                              json={"model": model, "max_tokens": max_tokens, "temperature": 0,
                                    "messages": [{"role": "system", "content": system},
                                                 {"role": "user", "content": user}]}).json()
            content = (r.get("choices") or [{}])[0].get("message", {}).get("content")
            if not content:
                last_err = r.get("error") or "empty reply"
                continue
            out = {"json": _extract_json(content), "model": model, "latency_s": round(time.time() - t0, 2),
                   "usage": r.get("usage", {})}
            db().llm_cache.insert_one({"key": key, "tier": tier, "response": out, "ts": time.time()})
            return out
        except (ValueError, requests.RequestException) as e:
            last_err = str(e)
    raise RuntimeError(f"all models failed: {last_err}")
