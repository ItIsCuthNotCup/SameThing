"""Metrics with product-grouped bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import DIFFERENT, MATCH, REVIEW


def pair_metrics(df: pd.DataFrame, decision: pd.Series, n_gold_total: int) -> dict:
    """df: candidate pairs with 'label'. n_gold_total: gold pairs in the split (incl. retrieval misses)."""
    y = df["label"].astype(int).values
    d = decision.values
    auto = d != REVIEW
    pm = d == MATCH
    tp = int((pm & (y == 1)).sum())
    fp = int((pm & (y == 0)).sum())
    fn_pairs = int(((~pm) & (y == 1)).sum())  # gold pairs in pool not auto-matched
    fn_e2e = n_gold_total - tp
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec_pool = tp / max(1, int((y == 1).sum()))
    rec_e2e = tp / max(1, n_gold_total)
    f1 = 2 * prec * rec_e2e / (prec + rec_e2e) if (tp + fp) and (prec + rec_e2e) else 0.0
    auto_acc = float(((d[auto] == MATCH) == (y[auto] == 1)).mean()) if auto.any() else float("nan")
    return {"pairs": len(df), "tp": tp, "fp": fp, "fn_pool": fn_pairs, "fn_e2e": fn_e2e,
            "precision": prec, "recall_pool": rec_pool, "recall_e2e": rec_e2e, "f1_e2e": f1,
            "review_fraction": float((~auto).mean()), "review_count": int((~auto).sum()),
            "auto_coverage": float(auto.mean()), "auto_accuracy": auto_acc}


def grouped_bootstrap_ci(df: pd.DataFrame, decision: pd.Series, n_gold_total: int, group_col: str = "left_rid",
                         n_boot: int = 500, seed: int = 0, keys=("precision", "recall_e2e", "review_fraction")) -> dict:
    """Resample by left record (proxy for underlying product); returns 95% percentile intervals."""
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    gidx = {g: np.where(df[group_col].values == g)[0] for g in groups}
    gold_per_group = df.groupby(group_col)["label"].sum().to_dict()
    # gold pairs outside the pool are attributed to left records with no pooled gold; approximate via scale factor
    pooled_gold = int(df["label"].sum())
    scale = n_gold_total / max(1, pooled_gold)
    out = {k: [] for k in keys}
    dvals = decision.values
    for _ in range(n_boot):
        sample = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([gidx[g] for g in sample])
        sub = df.iloc[idx]
        m = pair_metrics(sub, pd.Series(dvals[idx]), int(round(sub["label"].sum() * scale)))
        for k in keys:
            out[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}


def choose_thresholds(score: pd.Series, y: pd.Series, target_precision: float = 0.99,
                      lo_target_miss_rate: float = 0.01) -> tuple[float, float]:
    """Pick hi so that dev precision >= target (maximizing recall), lo so that <= miss rate of gold below it."""
    s, yy = score.values, y.values.astype(int)
    order = np.argsort(-s)
    best_hi, best_rec = 1.01, -1
    tp = fp = 0
    n_pos = yy.sum()
    for i in order:
        if np.isnan(s[i]):
            continue
        if yy[i]:
            tp += 1
        else:
            fp += 1
        if tp / (tp + fp) >= target_precision and tp / max(1, n_pos) > best_rec:
            best_rec, best_hi = tp / max(1, n_pos), s[i]
    pos_scores = np.sort(s[(yy == 1) & ~np.isnan(s)])
    lo = float(pos_scores[int(len(pos_scores) * lo_target_miss_rate)]) if len(pos_scores) else 0.0
    lo = min(lo, best_hi)
    return float(best_hi), float(lo)


def error_breakdown(df: pd.DataFrame, decision: pd.Series) -> pd.DataFrame:
    """Categorize false matches and missed matches by recorded material-difference type."""
    y = df["label"].astype(int)
    fp = df[(decision == MATCH) & (y == 0)]
    fn = df[(decision != MATCH) & (y == 1)]
    def cat(row):
        if row["capacity_conflict"]:
            return "capacity/variant"
        if row["accessory_mismatch"]:
            return "accessory vs product"
        if row["qty_conflict"]:
            return "quantity/multipack"
        if row["condition_conflict"]:
            return "condition"
        if row["bundle_mismatch"]:
            return "bundle"
        if row["model_conflict"]:
            return "model number differs"
        if row["model_unknown"]:
            return "identifier missing"
        return "wording"
    rows = []
    for name, sub in (("false_match", fp), ("missed_match", fn)):
        if len(sub):
            for k, v in sub.apply(cat, axis=1).value_counts().items():
                rows.append({"error": name, "category": k, "count": int(v)})
    return pd.DataFrame(rows)
