"""Timed test of Agent Engine memory (user_id = body type). Usage: PYTHONPATH=. uv run python scripts/test_agentengine_memory.py"""
import time

from engine import memory

ae = memory.backend()
print("backend:", memory.MEMORY_BACKEND)
if ae is None:
    raise SystemExit("Agent Engine memory not reachable")
from agent_engine_sdk_memory import MemoryRequestContext

s = ae.bind(MemoryRequestContext(user_id="drone", session_id="morph-test"))


def timed(name, fn):
    t = time.time()
    try:
        r = fn()
        print(f"{name:22s} OK   {1000 * (time.time() - t):6.0f} ms  {str(r)[:110]}")
    except Exception as e:
        print(f"{name:22s} FAIL {1000 * (time.time() - t):6.0f} ms  {type(e).__name__}: {str(e)[:110]}")


timed("save_episode", lambda: s.save_episode(title="v1 mission 903", content="FAIL drone v1: emitted move_to; body cannot move on ground"))
timed("search_episodes", lambda: s.search_episodes("drone cannot move on ground", top_k=3))
timed("save_procedure", lambda: s.save_procedure(procedure="drone-fly-every-leg", description="lesson from v17",
                                                 content="Use fly_to for every leg incl. charger; charge after pickup on long routes"))
timed("discover_procedures", lambda: s.discover_procedures("how should a drone move"))
timed("save_semantic", lambda: s.save_semantic(label="drone-specs", text="drone: 60 Wh battery, 1.0 kg payload, flight only"))
timed("search_semantic", lambda: s.search_semantic("drone battery capacity"))
timed("save_taxonomic", lambda: s.save_taxonomic(domain="bodies", term="drone",
                                                 definition="aerial body: flies, ignores ground terrain, low payload, short battery"))
timed("search_taxonomic", lambda: s.search_taxonomic("aerial body that flies"))
