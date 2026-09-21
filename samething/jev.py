"""Jev (TypeSafe System One) client with cache, budget, timeouts and bounded retries.

The API key is read server-side from TYPESAFE_API_KEY and never logged or cached.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

from .normalize import Record

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"  # pinned (alias jev-latest -> jev-1.13.0 on 2026-09-20)
PRICE_PER_MTOK_USD = 0.042  # input tokens only; output tokens free (docs.typesafe.ai/models, 2026-09-20)

MATCHING_DEFINITION = (
    "Two listings are the SAME PRODUCT when they refer to the same manufacturer model "
    "(same model number / variant). Different capacity, size, colour-coded model suffix, "
    "or a different model number means DIFFERENT. An accessory (case, cable, battery, mount) "
    "is DIFFERENT from the device it fits. If the visible text does not identify the model "
    "well enough to decide, answer insufficient."
)

# Exactly three prompt variants are permitted; the one used is recorded in the cache key.
QUESTION_SETS = {
    "v1": {
        "identifiers": {
            "type": "choice",
            "instructions": "Compare the model numbers / product identifiers visible in `a` and `b` (title, model, description). Do they establish agreement, conflict, or is there insufficient identifier information?",
            "criteria": {
                "agree": "Both listings show the same model number (ignoring punctuation/spacing).",
                "conflict": "Both listings show model numbers and they differ in a way that indicates a different model or variant (e.g. different suffix, capacity code).",
                "insufficient": "At least one listing shows no usable model number.",
            },
        },
        "same_model": {
            "type": "choice",
            "instructions": "Do the titles and descriptions of `a` and `b` identify the same manufacturer model of product?",
            "criteria": {
                "same": "Same brand and same specific model.",
                "different": "Different model, different variant, or different product type.",
                "insufficient": "Text is too vague to determine the specific model.",
            },
        },
        "accessory": {
            "type": "choice",
            "instructions": "Is one of `a` or `b` an accessory, part, or add-on (case, cable, battery, mount, remote, filter...) while the other is the main device?",
            "criteria": {
                "a_accessory": "`a` is the accessory, `b` is the main product.",
                "b_accessory": "`b` is the accessory, `a` is the main product.",
                "neither": "Both are main products or both are accessories of the same kind.",
            },
        },
        "variant_conflict": {
            "type": "choice",
            "instructions": "Is there an explicit conflict in capacity, size, screen size, wattage, colour, or region between `a` and `b`?",
            "criteria": {
                "conflict": "Both state a value for the same attribute and the values differ.",
                "no_conflict": "Stated values agree.",
                "not_stated": "At least one listing does not state the attribute.",
            },
        },
        "offer_diff": {
            "type": "choice",
            "instructions": "Do `a` and `b` differ in quantity (multipack vs single), condition (new vs refurbished/used), or bundle contents (kit vs base item)?",
            "criteria": {
                "differ": "An explicit difference in quantity, condition or bundle contents is stated.",
                "same": "Both state equivalent quantity/condition/contents.",
                "not_stated": "Not enough information to compare offers.",
            },
        },
        "verdict": {
            "type": "choice",
            "instructions": {"definition": MATCHING_DEFINITION,
                             "question": "Under `definition`, are `a` and `b` the same product?"},
            "criteria": {
                "same": "Same product under the definition.",
                "different": "Different product under the definition.",
                "insufficient": "The visible text does not allow a decision.",
            },
        },
    },
}


def _cache_key(model: str, variant: str, state: dict) -> str:
    payload = json.dumps({"m": model, "v": variant, "q": QUESTION_SETS[variant], "s": state}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class Budget:
    max_attempts: int = 3000
    max_usd: float = 3.0
    attempts: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    failures: int = 0
    latencies: list = None

    def __post_init__(self):
        self.latencies = []
        self._lock = threading.Lock()

    @property
    def usd(self) -> float:
        return self.input_tokens / 1e6 * PRICE_PER_MTOK_USD

    def exhausted(self) -> bool:
        return self.attempts >= self.max_attempts or self.usd >= self.max_usd

    def summary(self) -> dict:
        lat = sorted(self.latencies)
        return {"attempts": self.attempts, "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "estimated_usd": round(self.usd, 5), "failures": self.failures,
                "latency_p50_s": lat[len(lat) // 2] if lat else None,
                "latency_p95_s": lat[int(len(lat) * 0.95)] if lat else None}


class JevClient:
    def __init__(self, cache_path: str, budget: Budget | None = None, variant: str = "v1",
                 model: str = MODEL, timeout: float = 30.0, max_retries: int = 2, concurrency: int = 4,
                 api_key: str | None = None):
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        self.budget = budget or Budget()
        self.variant = variant
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.concurrency = concurrency
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("Jev")
        self._lock = threading.Lock()
        self._db = sqlite3.connect(cache_path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, variant TEXT, model TEXT, state TEXT, response TEXT, latency REAL, ts REAL)")
        self._db.commit()

    @property
    def available(self) -> bool:
        return bool(self._api_key)

    @staticmethod
    def make_state(a: Record, b: Record) -> dict:
        return {"a": a.visible_fields(), "b": b.visible_fields()}

    def _get_cached(self, key):
        with self._lock:
            row = self._db.execute("SELECT response, latency FROM cache WHERE key=?", (key,)).fetchone()
        if row:
            r = json.loads(row[0])
            r["_cached"] = True
            r["_latency"] = row[1]
            return r
        return None

    def _put(self, key, state, response, latency):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?)",
                             (key, self.variant, self.model, json.dumps(state, sort_keys=True), json.dumps(response), latency, time.time()))
            self._db.commit()

    def judge(self, a: Record, b: Record) -> dict:
        """Return {'answers':..., 'model':..., 'usage':..., '_latency':..., '_error':...}."""
        state = self.make_state(a, b)
        key = _cache_key(self.model, self.variant, state)
        cached = self._get_cached(key)
        if cached:
            return cached
        if not self.available:
            return {"_error": "no_api_key", "answers": None}
        body = {"state": state, "model": self.model, "questions": QUESTION_SETS[self.variant]}
        last_err = None
        for attempt in range(self.max_retries + 1):
            with self.budget._lock:
                if self.budget.exhausted():
                    return {"_error": "budget_exhausted", "answers": None}
                self.budget.attempts += 1
            t0 = time.time()
            try:
                resp = requests.post(API_URL, json=body, timeout=self.timeout,
                                     headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"})
                lat = time.time() - t0
                if resp.status_code == 200:
                    data = resp.json()
                    with self.budget._lock:
                        self.budget.input_tokens += data.get("usage", {}).get("input_tokens", 0)
                        self.budget.output_tokens += data.get("usage", {}).get("output_tokens", 0)
                        self.budget.latencies.append(lat)
                    data["_latency"] = lat
                    data["_cached"] = False
                    self._put(key, state, data, lat)
                    return data
                last_err = f"http_{resp.status_code}"
                if resp.status_code in (429, 529, 500, 502, 503):
                    time.sleep(min(8, 1.5 * (2 ** attempt)))
                    continue
                if resp.status_code == 422:
                    last_err += ":" + resp.text[:200]
                break
            except requests.RequestException as e:
                last_err = f"exception:{type(e).__name__}"
                time.sleep(min(8, 1.5 * (2 ** attempt)))
        with self.budget._lock:
            self.budget.failures += 1
        return {"_error": last_err, "answers": None}

    def judge_many(self, pairs: list[tuple[Record, Record]]) -> list[dict]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(lambda p: self.judge(*p), pairs))

    def cache_size(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]


def summarize_answers(resp: dict) -> dict:
    """Flatten a Jev response into decision-relevant fields (no invented reasoning)."""
    out = {"jev_error": resp.get("_error"), "jev_model": resp.get("model"), "jev_latency": resp.get("_latency"),
           "jev_cached": resp.get("_cached"), "jev_input_tokens": (resp.get("usage") or {}).get("input_tokens")}
    ans = resp.get("answers") or {}
    for q, a in ans.items():
        out[f"jev_{q}"] = a.get("choice")
        out[f"jev_{q}_conf"] = a.get("confidence")
        for opt, p in (a.get("probabilities") or {}).items():
            out[f"jev_{q}_p_{opt}"] = p
    return out
