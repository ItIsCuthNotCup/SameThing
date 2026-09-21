"""Pairwise features and deterministic rules. Every method sees the same visible fields."""
from __future__ import annotations

from rapidfuzz import fuzz

from .normalize import Record, tokens

MATCH, DIFFERENT, REVIEW = "match", "different", "review"


def _model_relation(a: Record, b: Record) -> tuple[str, str]:
    """Return (relation, detail): agree | conflict | prefix | unknown."""
    A, B = set(a.model_tokens), set(b.model_tokens)
    if not A or not B:
        return "unknown", "one side has no model-like token in its title"
    shared = A & B
    # variant: a token on one side extends a shared/other token on the other side (dvpns57p vs dvpns57ps)
    for x in A - B:
        for y in B - A:
            if len(x) >= 5 and len(y) >= 5 and (x.startswith(y) or y.startswith(x)) and abs(len(x) - len(y)) <= 3:
                if shared:
                    return "variant", f"shared {', '.join(sorted(shared))} but {x} vs {y} differ by suffix"
    if shared:
        return "agree", f"shared model token(s): {', '.join(sorted(shared))}"
    # prefix relation (e.g. 'pslx350' vs 'pslx350h') is weak agreement
    for x in A:
        for y in B:
            if len(x) >= 5 and len(y) >= 5 and (x.startswith(y) or y.startswith(x)):
                if abs(len(x) - len(y)) <= 2:
                    return "variant", f"{x} vs {y} differ only by suffix"
                return "prefix", f"{x} vs {y} share a prefix"
    # explicit conflicting identifiers: both have tokens with digits, none shared
    ax = a.explicit_model or (a.model_tokens[0] if a.model_tokens else "")
    bx = b.explicit_model or (b.model_tokens[0] if b.model_tokens else "")
    return "conflict", f"no shared model token ({ax} vs {bx})"


def pair_features(a: Record, b: Record) -> dict:
    f: dict = {}
    f["title_ratio"] = fuzz.ratio(a.title_norm, b.title_norm) / 100
    f["title_token_set"] = fuzz.token_set_ratio(a.title_norm, b.title_norm) / 100
    f["title_token_sort"] = fuzz.token_sort_ratio(a.title_norm, b.title_norm) / 100
    f["title_partial"] = fuzz.partial_ratio(a.title_norm, b.title_norm) / 100
    ta, tb = set(tokens(a.title_norm)), set(tokens(b.title_norm))
    f["title_jaccard"] = len(ta & tb) / max(1, len(ta | tb))
    da = set(tokens(f"{a.title_norm} {a.desc_norm}"))
    db = set(tokens(f"{b.title_norm} {b.desc_norm}"))
    f["full_jaccard"] = len(da & db) / max(1, len(da | db))
    f["desc_token_set"] = fuzz.token_set_ratio(a.desc_norm[:400], b.desc_norm[:400]) / 100 if a.desc_norm and b.desc_norm else 0.0
    rel, detail = _model_relation(a, b)
    f["model_relation"] = rel
    f["model_detail"] = detail
    f["model_agree"] = 1.0 if rel == "agree" else 0.0
    f["model_conflict"] = 1.0 if rel == "conflict" else 0.0
    f["model_prefix"] = 1.0 if rel == "prefix" else 0.0
    f["model_unknown"] = 1.0 if rel == "unknown" else 0.0
    f["model_variant"] = 1.0 if rel == "variant" else 0.0
    da_, db_ = set(a.desc_model_tokens) | set(a.model_tokens), set(b.desc_model_tokens) | set(b.model_tokens)
    f["desc_model_agree"] = 1.0 if (da_ & db_) else 0.0
    f["compat_ref"] = 1.0 if (set(a.compat_model_tokens) & set(b.model_tokens)) or (set(b.compat_model_tokens) & set(a.model_tokens)) else 0.0
    # brand
    if a.brand_norm and b.brand_norm:
        f["brand_same"] = 1.0 if (a.brand_norm == b.brand_norm or a.brand_norm in b.title_norm or b.brand_norm in a.title_norm) else 0.0
    else:
        f["brand_same"] = 0.5
    # numbers present in one title but not the other (variant signal)
    na = {t for t in tokens(a.title_norm) if any(c.isdigit() for c in t)}
    nb = {t for t in tokens(b.title_norm) if any(c.isdigit() for c in t)}
    f["num_overlap"] = len(na & nb) / max(1, len(na | nb)) if (na or nb) else 1.0
    # capacity conflict: same unit, disjoint values
    cap_conf = []
    for unit, va in a.capacities.items():
        vb = b.capacities.get(unit)
        if vb and not set(va) & set(vb):
            cap_conf.append(f"{unit}: {va} vs {vb}")
    f["capacity_conflict"] = 1.0 if cap_conf else 0.0
    f["capacity_detail"] = "; ".join(cap_conf)
    # quantity / condition / accessory / bundle
    qa, qb = a.pack_qty or 1, b.pack_qty or 1
    f["qty_conflict"] = 1.0 if qa != qb else 0.0
    f["qty_detail"] = f"{qa} vs {qb}" if qa != qb else ""
    f["condition_conflict"] = 1.0 if (a.condition != b.condition and "unknown" not in (a.condition, b.condition)) else 0.0
    f["condition_detail"] = f"{a.condition} vs {b.condition}" if a.condition != b.condition else ""
    f["accessory_mismatch"] = 1.0 if a.accessory_hint != b.accessory_hint else 0.0
    f["bundle_mismatch"] = 1.0 if a.bundle_hint != b.bundle_hint else 0.0
    # price ratio when both known & same currency
    if a.price and b.price and a.currency == b.currency:
        f["price_ratio"] = min(a.price, b.price) / max(a.price, b.price)
    else:
        f["price_ratio"] = -1.0
    return f


NUMERIC_FEATURES = [
    "title_ratio", "title_token_set", "title_token_sort", "title_partial", "title_jaccard",
    "full_jaccard", "desc_token_set", "model_agree", "model_conflict", "model_prefix", "model_unknown",
    "model_variant", "desc_model_agree", "compat_ref",
    "brand_same", "num_overlap", "capacity_conflict", "qty_conflict", "condition_conflict",
    "accessory_mismatch", "bundle_mismatch", "price_ratio",
]


def fuzzy_score(f: dict) -> float:
    """Strong fuzzy score: weighted string similarity with identifier/attribute adjustments."""
    s = 0.45 * f["title_token_set"] + 0.25 * f["title_ratio"] + 0.15 * f["title_jaccard"] + 0.15 * f["num_overlap"]
    if f["model_agree"]:
        s += 0.25
    elif f["model_prefix"]:
        s += 0.10
    elif f["model_variant"]:
        s -= 0.15
    elif f["model_conflict"]:
        s -= 0.25
    elif f["desc_model_agree"]:
        s += 0.10
    if f["compat_ref"]:
        s -= 0.3
    if f["brand_same"] == 0.0:
        s -= 0.3
    if f["capacity_conflict"]:
        s -= 0.3
    return max(0.0, min(1.0, s))


def material_differences(f: dict) -> list[str]:
    diffs = []
    if f["model_variant"]:
        diffs.append(f"possible model variant ({f['model_detail']})")
    if f["compat_ref"]:
        diffs.append("one title references the other's model as a compatibility target")
    if f["capacity_conflict"]:
        diffs.append(f"capacity conflict ({f['capacity_detail']})")
    if f["qty_conflict"]:
        diffs.append(f"quantity differs ({f['qty_detail']})")
    if f["condition_conflict"]:
        diffs.append(f"condition differs ({f['condition_detail']})")
    if f["accessory_mismatch"]:
        diffs.append("one title looks like an accessory")
    if f["bundle_mismatch"]:
        diffs.append("one title looks like a bundle/kit")
    return diffs


def apply_rules(f: dict) -> tuple[str, str]:
    """Deterministic resolution of clear cases. Returns (decision, reason) with REVIEW meaning 'unresolved'."""
    if f["brand_same"] == 0.0 and f["model_agree"] == 0.0:
        return DIFFERENT, "rule: brand mismatch without identifier agreement"
    if f["capacity_conflict"]:
        return DIFFERENT, f"rule: explicit capacity conflict ({f['capacity_detail']})"
    if f["compat_ref"]:
        return DIFFERENT, "rule: one title cites the other's model only as a compatibility reference (accessory/part)"
    if f["model_variant"]:
        return REVIEW, f"rule: possible variant ({f['model_detail']})"
    if f["model_agree"] and f["title_token_set"] >= 0.6 and not f["qty_conflict"] and not f["condition_conflict"] and not f["accessory_mismatch"]:
        return MATCH, f"rule: identifier agreement ({f['model_detail']}) with similar titles"
    if f["model_conflict"] and f["title_token_set"] < 0.75:
        return DIFFERENT, f"rule: identifier conflict ({f['model_detail']}) with dissimilar titles"
    if f["title_token_set"] < 0.45 and f["title_ratio"] < 0.4:
        return DIFFERENT, "rule: titles share little text"
    return REVIEW, "rules did not resolve"
