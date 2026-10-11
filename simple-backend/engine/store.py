"""MongoDB Atlas storage for Morph: collections, indexes, helpers."""
from __future__ import annotations

import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.database import Database
from pymongo.errors import CollectionInvalid, OperationFailure

load_dotenv()
DB_NAME = os.getenv("MORPH_DB", "morph")
MEMORY_KINDS = ("semantic", "episodic", "procedural", "taxonomic")


@lru_cache(maxsize=1)
def client() -> MongoClient:
    return MongoClient(os.environ["MONGODB_URI"], appname="morph")


def db() -> Database:
    return client()[DB_NAME]


def _search_index(coll, name: str, definition: dict, kind: str | None = None):
    existing = {i["name"] for i in coll.list_search_indexes()}
    if name in existing:
        return
    model = {"name": name, "definition": definition}
    if kind:
        model["type"] = kind
    coll.create_search_index(model)


def ensure_schema() -> dict:
    """Idempotent: create collections and indexes. Returns what exists."""
    d = db()
    try:
        d.create_collection("metrics", timeseries={"timeField": "ts", "metaField": "meta", "granularity": "seconds"})
    except (CollectionInvalid, OperationFailure):
        pass
    for name in ("harnesses", "trials", "evaluations", "failures", "checkpoints", "llm_cache", "frames",
                 *MEMORY_KINDS):
        if name not in d.list_collection_names():
            d.create_collection(name)

    d.harnesses.create_index([("body_type", ASCENDING), ("version", ASCENDING)], unique=True)
    d.trials.create_index([("harness_id", ASCENDING), ("mission_id", ASCENDING)])
    d.trials.create_index([("body", ASCENDING), ("split", ASCENDING), ("ts", DESCENDING)])
    d.evaluations.create_index([("candidate_id", ASCENDING)], unique=True)
    d.checkpoints.create_index([("run_id", ASCENDING)], unique=True)
    d.llm_cache.create_index([("key", ASCENDING)], unique=True)
    d.taxonomic.create_index([("domain", ASCENDING), ("term", ASCENDING)], unique=True)
    d.procedural.create_index([("user_id", ASCENDING), ("procedure", ASCENDING)], unique=True)
    d.semantic.create_index([("user_id", ASCENDING), ("label", ASCENDING)], unique=True)

    # failure retrieval: autoEmbed vector index + keyword index for $rankFusion
    _search_index(d.failures, "failures_vec",
                  {"fields": [{"type": "autoEmbed", "modality": "text", "path": "summary", "model": "voyage-4-lite"},
                              {"type": "filter", "path": "body"}]}, "vectorSearch")
    _search_index(d.failures, "failures_text", {"mappings": {"dynamic": False, "fields": {
        "summary": {"type": "string"}, "body": {"type": "token"}}}})
    for kind, path in (("episodic", "content"), ("procedural", "content"), ("semantic", "text"),
                       ("taxonomic", "definition")):
        _search_index(d[kind], f"{kind}_vec",
                      {"fields": [{"type": "autoEmbed", "modality": "text", "path": path, "model": "voyage-4-lite"}]},
                      "vectorSearch")
    return {"collections": sorted(d.list_collection_names())}


def wait_search_ready(timeout_s: int = 180) -> dict:
    d, t0 = db(), time.time()
    while True:
        status = {f"{c}.{i['name']}": i.get("status") for c in ("failures", *MEMORY_KINDS)
                  for i in d[c].list_search_indexes()}
        if all(s == "READY" for s in status.values()) or time.time() - t0 > timeout_s:
            return status
        time.sleep(5)
