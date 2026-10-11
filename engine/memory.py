"""Four-type agent memory with the Agent Engine API shape, backed by Atlas collections.

user_id = body category (e.g. "drone"). Taxonomic memory is shared across bodies.
Search uses autoEmbed vector indexes (Voyage, embedded server-side by Atlas).
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


class Memory:
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
