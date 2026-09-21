"""End-to-end matching pipeline producing per-pair scores and three-way decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .candidates import retrieve_candidates
from .features import (DIFFERENT, MATCH, NUMERIC_FEATURES, REVIEW, apply_rules, fuzzy_score,
                       material_differences, pair_features)
from .jev import JevClient, summarize_answers
from .normalize import Record, prune_common_model_tokens

METHODS = ["exact", "fuzzy", "supervised", "jev", "hybrid"]


def build_pairs(left: list[Record], right: list[Record], k: int = 8) -> pd.DataFrame:
    pruned = prune_common_model_tokens([left, right])
    cands = retrieve_candidates(left, right, k=k)
    rindex = {r.rid: r for r in right}
    rows = []
    for l in left:
        for rank, c in enumerate(cands.get(l.rid, [])):
            r = rindex[c["right_rid"]]
            f = pair_features(l, r)
            rule, reason = apply_rules(f)
            row = {"left_rid": l.rid, "right_rid": r.rid, "rank": rank, "methods": "|".join(c["methods"]),
                   "tfidf_char": c["tfidf_char"], "tfidf_tok": c["tfidf_tok"],
                   "rule_decision": rule, "rule_reason": reason, "fuzzy_score": fuzzy_score(f),
                   "material_diffs": "; ".join(material_differences(f)),
                   "exact_title": 1.0 if l.title_norm == r.title_norm else 0.0}
            row.update(f)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["pruned_model_tokens"] = pruned
    return df, cands


# --------------------------------------------------------------------------
# Method scores -> three-way decision via (hi, lo) thresholds
# --------------------------------------------------------------------------

def exact_decision(df: pd.DataFrame) -> pd.Series:
    """A. exact normalized matching: identical normalized title or shared explicit model token >=5 chars."""
    strong_model = df["model_agree"].astype(bool) & df["model_detail"].str.extract(r"shared model token\(s\): (\S+)")[0].fillna("").str.len().ge(5)
    is_match = df["exact_title"].astype(bool) | strong_model
    return pd.Series(np.where(is_match, MATCH, DIFFERENT), index=df.index)


def threshold_decision(score: pd.Series, hi: float, lo: float) -> pd.Series:
    return pd.Series(np.select([score >= hi, score < lo], [MATCH, DIFFERENT], REVIEW), index=score.index)


def fuzzy_with_rules(df: pd.DataFrame, hi: float, lo: float) -> pd.Series:
    """B. strong fuzzy matching + identifier/attribute conflict rules (rules resolve first, fuzzy score for the rest)."""
    d = threshold_decision(df["fuzzy_score"], hi, lo)
    resolved = df["rule_decision"] != REVIEW
    d[resolved] = df.loc[resolved, "rule_decision"]
    hard_no = (df["capacity_conflict"] == 1) | ((df["brand_same"] == 0) & (df["model_agree"] == 0))
    d[hard_no] = DIFFERENT
    # offer-level conflicts should never be auto-matched
    offer = (df["qty_conflict"] == 1) | (df["condition_conflict"] == 1)
    d[offer & (d == MATCH)] = REVIEW
    return d


def train_supervised(dev: pd.DataFrame):
    X = dev[NUMERIC_FEATURES].values
    y = dev["label"].values
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
    oof = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    clf.fit(X, y)
    return clf, oof


def supervised_scores(clf, df: pd.DataFrame) -> pd.Series:
    return pd.Series(clf.predict_proba(df[NUMERIC_FEATURES].values)[:, 1], index=df.index)


def jev_pair_score(row: pd.Series) -> float:
    """Combine Jev answers into one P(same) signal. Conflicting evidence is preserved in columns.

    Score = P(verdict=same) reduced by explicit accessory / variant-conflict signals.
    """
    p = row.get("jev_verdict_p_same")
    if p is None or pd.isna(p):
        return np.nan
    p = float(p)
    p_conf = float(row.get("jev_identifiers_p_conflict", 0) or 0)
    p_var = float(row.get("jev_variant_conflict_p_conflict", 0) or 0)
    p_acc = 1.0 - float(row.get("jev_accessory_p_neither", 1) or 1)
    penalty = max(p_conf, p_var, p_acc)
    return p * (1.0 - 0.5 * penalty)


def run_jev(df: pd.DataFrame, left: list[Record], right: list[Record], client: JevClient, mask: pd.Series) -> pd.DataFrame:
    """Ask Jev about the rows where mask is True. Adds jev_* columns. Failures stay NaN (-> review)."""
    li = {r.rid: r for r in left}
    ri = {r.rid: r for r in right}
    sub = df[mask]
    pairs = [(li[a], ri[b]) for a, b in zip(sub["left_rid"], sub["right_rid"])]
    responses = client.judge_many(pairs)
    recs = [summarize_answers(r) for r in responses]
    jdf = pd.DataFrame(recs, index=sub.index)
    for c in jdf.columns:
        df.loc[sub.index, c] = jdf[c]
    df["jev_score"] = df.apply(jev_pair_score, axis=1) if "jev_verdict_p_same" in df.columns else np.nan
    return df


def jev_decision(df: pd.DataFrame, hi: float, lo: float) -> pd.Series:
    """D. Jev alone on the pool; pairs without a Jev answer -> review."""
    s = df["jev_score"]
    d = threshold_decision(s.fillna(-1), hi, lo)
    d[s.isna()] = REVIEW
    # Jev-reported offer differences are surfaced as review, never auto-match
    if "jev_offer_diff" in df.columns:
        d[(df["jev_offer_diff"] == "differ") & (d == MATCH)] = REVIEW
    return d


def hybrid_decision(df: pd.DataFrame, hi: float, lo: float) -> pd.Series:
    """E. rules first; Jev for pairs the rules left unresolved."""
    d = df["rule_decision"].copy()
    unresolved = d == REVIEW
    jd = jev_decision(df, hi, lo)
    d[unresolved] = jd[unresolved]
    return d


def explain_pair(row: pd.Series) -> list[str]:
    """Human-readable evidence assembled only from recorded comparisons/answers."""
    ev = [f"retrieved by: {row['methods']}", f"title similarity (token set): {row['title_token_set']:.2f}",
          f"model tokens: {row['model_detail']}",
          f"brand: {'same' if row['brand_same'] == 1 else 'different' if row['brand_same'] == 0 else 'unknown'}",
          f"rule outcome: {row['rule_reason']}"]
    if row.get("material_diffs"):
        ev.append(f"material differences: {row['material_diffs']}")
    if isinstance(row.get("jev_verdict"), str):
        ev.append(f"Jev verdict: {row['jev_verdict']} (p_same={row.get('jev_verdict_p_same', float('nan')):.2f}, conf={row.get('jev_verdict_conf', float('nan')):.2f})")
        for q in ("identifiers", "same_model", "accessory", "variant_conflict", "offer_diff"):
            v = row.get(f"jev_{q}")
            if isinstance(v, str):
                ev.append(f"Jev {q}: {v} (conf={row.get(f'jev_{q}_conf', float('nan')):.2f})")
    elif isinstance(row.get("jev_error"), str):
        ev.append(f"Jev not available for this pair: {row['jev_error']}")
    return ev
