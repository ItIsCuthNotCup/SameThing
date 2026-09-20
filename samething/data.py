"""Benchmark loading and group-aware splitting."""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import defaultdict

import pandas as pd

from .normalize import ColumnMap, build_records

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")

BENCHMARKS = {
    "abt-buy": {
        "left": os.path.join(RAW, "abt-buy", "Abt.csv"),
        "right": os.path.join(RAW, "abt-buy", "Buy.csv"),
        "mapping": os.path.join(RAW, "abt-buy", "abt_buy_perfectMapping.csv"),
        "map_cols": ("idAbt", "idBuy"),
        "left_map": ColumnMap(id="id", title="name", description="description", price="price", default_currency="USD"),
        "right_map": ColumnMap(id="id", title="name", description="description", brand="manufacturer", price="price", default_currency="USD"),
        "left_name": "Abt", "right_name": "Buy",
    },
    "amazon-google": {
        "left": os.path.join(RAW, "amazon-google", "Amazon.csv"),
        "right": os.path.join(RAW, "amazon-google", "GoogleProducts.csv"),
        "mapping": os.path.join(RAW, "amazon-google", "Amzon_GoogleProducts_perfectMapping.csv"),
        "map_cols": ("idAmazon", "idGoogleBase"),
        "left_map": ColumnMap(id="id", title="title", description="description", brand="manufacturer", price="price", default_currency="USD"),
        "right_map": ColumnMap(id="id", title="name", description="description", brand="manufacturer", price="price", default_currency="USD"),
        "left_name": "Amazon", "right_name": "Google",
    },
}


def read_csv(path) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1"):
        try:
            if hasattr(path, "seek"):
                path.seek(0)
            return pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise


def load_benchmark(name: str):
    b = BENCHMARKS[name]
    ldf, rdf = read_csv(b["left"]), read_csv(b["right"])
    left = build_records(ldf, b["left_map"], b["left_name"])
    right = build_records(rdf, b["right_map"], b["right_name"])
    mdf = read_csv(b["mapping"])
    gold = {(str(a), str(c)) for a, c in zip(mdf[b["map_cols"][0]], mdf[b["map_cols"][1]])}
    return left, right, gold, ldf, rdf


def product_groups(left_ids, right_ids, gold):
    """Connected components over the gold mapping; unmatched records are singleton groups."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for l in left_ids:
        find(("L", l))
    for r in right_ids:
        find(("R", r))
    for l, r in gold:
        union(("L", l), ("R", r))
    groups = defaultdict(list)
    for node in list(parent):
        groups[find(node)].append(node)
    return list(groups.values())


def group_split(left, right, gold, dev_frac=0.4, seed=13):
    """Split records so that no underlying product spans dev and test.

    Every record (matched or not) is assigned to exactly one split.
    """
    groups = product_groups([r.rid for r in left], [r.rid for r in right], gold)
    groups.sort(key=lambda g: json.dumps(sorted(g)))
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_dev = int(len(groups) * dev_frac)
    split = {}
    for i, g in enumerate(groups):
        name = "dev" if i < n_dev else "test"
        for node in g:
            split[node] = name
    return split


def split_manifest(split, left, right, gold, out_path):
    rows = []
    for r in left:
        rows.append({"source": r.source, "rid": r.rid, "split": split[("L", r.rid)]})
    for r in right:
        rows.append({"source": r.source, "rid": r.rid, "split": split[("R", r.rid)]})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    stats = df.groupby(["split", "source"]).size().to_dict()
    gold_by_split = defaultdict(int)
    for l, r in gold:
        gold_by_split[split[("L", l)]] += 1
    return {"records": {f"{k[0]}/{k[1]}": v for k, v in stats.items()}, "gold_pairs": dict(gold_by_split)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()
