"""Four-type agent memory. Primary backend: MongoDB Atlas Agent Engine memory (agent-engine-sdk-memory,
service-account auth). Fallback: Atlas collections with the same API (autoEmbed vector indexes).

user_id = body category (e.g. "drone"). Taxonomic memory is shared across bodies.
MEMORY_BACKEND reports which backend is live ("agentengine" or "atlas-collections").
"""
from __future__ import annotations

import time

from engine.store import db


def _vsearch(coll: str, path: str, query: str, k: int, filt: dict | None = None) -> list[dict]:
    if k <= 0:
        return []
    stage = {"index": f"{coll}_vec", "path": path, "query": {"text": query}, "limit": k, "numCandidates": 50}
    docs = list(db()[coll].aggregate([{"$vectorSearch": stage}, *([{"$match": filt}] if filt else []),
                                      {"$project": {"_id": 0, "embedding": 0}},
                                      {"$set": {"score": {"$meta": "vectorSearchScore"}}}]))
    return docs


def _ae_client():
    """Agent Engine memory client via a project service account, or None if unavailable."""
    import os
    import requests
    from dotenv import dotenv_values
    env = {**dotenv_values(), **os.environ}
    if os.getenv("MORPH_MEMORY", "agentengine") != "agentengine" or not env.get("AE_CLIENT_ID"):
        return None
    try:
        from agent_engine_sdk_memory import Memory as AEMemory
        tok = requests.post("https://agentengine.mongodb.com/api/v1/oauth/token", timeout=20,
                            auth=(env["AE_CLIENT_ID"], env["AE_CLIENT_SECRET"]),
                            data={"grant_type": "client_credentials"}).json()["access_token"]
        # project_id selects the project-scoped routes (/api/v1/projects/{id}/memory/*) the hosted server serves
        ae = AEMemory(service_account_token=tok, project_id=env.get("AGENTIC_MEMORY_PROJECT_ID"))
        ae.list_domains()  # readiness probe: raises MemoryNotProvisionedError etc. until the server is up
        return ae
    except Exception as e:
        print(f"[memory] Agent Engine unavailable ({str(e)[:80]}); using Atlas collections")
        return None


_AE = None
MEMORY_BACKEND = "atlas-collections"


def backend():
    global _AE, MEMORY_BACKEND
    if _AE is None:
        _AE = _ae_client() or False
        MEMORY_BACKEND = "agentengine" if _AE else "atlas-collections"
    return _AE or None


class AgentEngineMemory:
    """Same API as Memory, delegating to the hosted Agent Engine memory server."""

    def __init__(self, ae, user_id: str, session_id: str = "default"):
        from agent_engine_sdk_memory import MemoryRequestContext
        self.s = ae.bind(MemoryRequestContext(user_id=user_id, session_id=session_id))

    def save_episode(self, title, content, **meta):
        return self.s.save_episode(title=title, content=content)

    def search_episodes(self, query, k=3):
        return [{"content": getattr(h, "content", str(h))} for h in self.s.search_episodes(query, top_k=k)]

    def save_procedure(self, procedure, description, content):
        return self.s.save_procedure(procedure=procedure, description=description, content=content)

    def discover_procedures(self, query, k=2):
        return [{"content": getattr(h, "content", str(h))} for h in self.s.discover_procedures(query)][:k]

    def save_semantic(self, label, text):
        return self.s.save_semantic(label=label, text=text)

    def search_semantic(self, query, k=3):
        return [{"text": getattr(h, "text", str(h))} for h in self.s.search_semantic(query)][:k]


class Memory:
    def __new__(cls, user_id: str, session_id: str = "default"):
        ae = backend()
        if ae is not None:
            return AgentEngineMemory(ae, user_id, session_id)
        return super().__new__(cls)

    def __init__(self, user_id: str, session_id: str = "default"):
        self.user_id, self.session_id = user_id, session_id

    # episodic
    def save_episode(self, title: str, content: str, **meta):
        db().episodic.insert_one({"user_id": self.user_id, "session_id": self.session_id, "title": title,
                                  "content": content, "ts": time.time(), **meta})

    def search_episodes(self, query: str, k: int = 3) -> list[dict]:
        return _vsearch("episodic", "content", query, k * 3, {"user_id": self.user_id})[:k]

    # procedural
    def save_procedure(self, procedure: str, description: str, content: str):
        db().procedural.update_one({"user_id": self.user_id, "procedure": procedure},
                                   {"$set": {"description": description, "content": content, "ts": time.time()}},
                                   upsert=True)

    def discover_procedures(self, query: str, k: int = 2) -> list[dict]:
        return _vsearch("procedural", "content", query, k * 3, {"user_id": self.user_id})[:k]

    # semantic
    def save_semantic(self, label: str, text: str):
        db().semantic.update_one({"user_id": self.user_id, "label": label},
                                 {"$set": {"text": text, "ts": time.time()}}, upsert=True)

    def search_semantic(self, query: str, k: int = 3) -> list[dict]:
        return _vsearch("semantic", "text", query, k * 3, {"user_id": self.user_id})[:k]

    # taxonomic (shared)
    @staticmethod
    def save_taxonomic(domain: str, term: str, definition: str):
        db().taxonomic.update_one({"domain": domain, "term": term},
                                  {"$set": {"definition": definition, "ts": time.time()}}, upsert=True)

    @staticmethod
    def search_taxonomic(query: str, k: int = 3) -> list[dict]:
        return _vsearch("taxonomic", "definition", query, k)
