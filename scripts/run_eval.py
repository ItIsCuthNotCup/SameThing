"""Full evaluation: dev tuning -> frozen thresholds -> held-out test.

Usage: python scripts/run_eval.py [--benchmark abt-buy] [--no-jev] [--dev-left 120] [--jev-topk 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samething import data as D  # noqa: E402
from samething.candidates import candidate_recall  # noqa: E402
from samething.evaluate import choose_thresholds, error_breakdown, grouped_bootstrap_ci, pair_metrics  # noqa: E402
from samething.features import MATCH, REVIEW  # noqa: E402
from samething.jev import Budget, JevClient  # noqa: E402
from samething.pipeline import (build_pairs, exact_decision, fuzzy_with_rules, hybrid_decision, jev_decision,
                                run_jev, supervised_scores, threshold_decision, train_supervised)  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--benchmark", default="abt-buy")
ap.add_argument("--no-jev", action="store_true")
ap.add_argument("--dev-left", type=int, default=120, help="left records in dev used for Jev tuning")
ap.add_argument("--jev-topk", type=int, default=3)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--max-attempts", type=int, default=2700)
ap.add_argument("--variant", default="v1")
ap.add_argument("--out", default=None)
args = ap.parse_args()

OUT = args.out or os.path.join(D.ROOT, "results", args.benchmark)
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

left, right, gold, _, _ = D.load_benchmark(args.benchmark)
split = D.group_split(left, right, gold)
manifest = D.split_manifest(split, left, right, gold, os.path.join(OUT, "split_manifest.csv"))
print("split", manifest)

# --- retrieval on the whole catalog (no labels used) --------------------------------------
pairs, cands = build_pairs(left, right, k=args.k)
gold_set = set(gold)
pairs["label"] = [int((a, b) in gold_set) for a, b in zip(pairs["left_rid"], pairs["right_rid"])]
pairs["split"] = [split[("L", a)] for a in pairs["left_rid"]]
pairs["in_jev_pool"] = pairs["rank"] < args.jev_topk

retr = {}
for name in ("dev", "test"):
    g = {(l, r) for l, r in gold if split[("L", l)] == name}
    c = {l: v for l, v in cands.items() if split[("L", l)] == name}
    cr = candidate_recall(c, g)
    crk = candidate_recall({l: v[: args.jev_topk] for l, v in c.items()}, g)
    retr[name] = {"gold": cr["gold"], f"recall@{2*args.k}": cr["recall"], "pairs": cr["n_candidate_pairs"],
                  f"recall@{args.jev_topk}(jev_pool)": crk["recall"], "misses": cr["misses"][:20]}
print("retrieval", json.dumps(retr, indent=1))

dev = pairs[pairs.split == "dev"].copy()
test = pairs[pairs.split == "test"].copy()
n_gold_dev = retr["dev"]["gold"]
n_gold_test = retr["test"]["gold"]

# --- Jev calls (dev subset for tuning, full test pool) -----------------------------------
jev_ran = False
budget = Budget(max_attempts=args.max_attempts)
client = JevClient(os.path.join(D.ROOT, "data", "cache", "jev_cache.sqlite"), budget=budget, variant=args.variant)
jev_status = {"ran": False, "reason": None}
if args.no_jev or not client.available:
    jev_status["reason"] = "disabled by flag" if args.no_jev else "TYPESAFE_API_KEY not set"
    print("Jev NOT run:", jev_status["reason"])
else:
    rng = np.random.default_rng(7)
    dev_left_ids = sorted(dev["left_rid"].unique())
    dev_sub = set(rng.choice(dev_left_ids, size=min(args.dev_left, len(dev_left_ids)), replace=False))
    dev_mask = dev["left_rid"].isin(dev_sub) & (dev["rank"] < args.jev_topk + 1)
    print(f"Jev dev calls: {int(dev_mask.sum())}")
    dev = run_jev(dev, left, right, client, dev_mask)
    print("budget after dev", budget.summary())
    test_mask = test["in_jev_pool"]
    print(f"Jev test calls: {int(test_mask.sum())}")
    test = run_jev(test, left, right, client, test_mask)
    print("budget after test", budget.summary())
    jev_ran = True
    jev_status = {"ran": True, "model": str(test["jev_model"].dropna().iloc[0]) if test["jev_model"].notna().any() else None,
                  "variant": args.variant, "budget": budget.summary(), "cache_entries": client.cache_size()}

if not jev_ran:
    for d in (dev, test):
        d["jev_score"] = np.nan

# --- tune on dev (frozen before test) ----------------------------------------------------
TARGETS = {"99": 0.99, "95": 0.95}
thr = {}
clf, oof = train_supervised(dev)
dev["sup_score"] = oof
for tag, tp in TARGETS.items():
    unres_dev = dev[dev["rule_decision"] == REVIEW]
    thr[f"fuzzy@{tag}"] = choose_thresholds(unres_dev["fuzzy_score"], unres_dev["label"], target_precision=tp)
    thr[f"supervised@{tag}"] = choose_thresholds(dev["sup_score"], dev["label"], target_precision=tp)
    if jev_ran:
        dj = dev[dev["jev_score"].notna()]
        thr[f"jev@{tag}"] = choose_thresholds(dj["jev_score"], dj["label"], target_precision=tp)
        unresolved = dj[dj["rule_decision"] == REVIEW]
        thr[f"hybrid@{tag}"] = choose_thresholds(unresolved["jev_score"], unresolved["label"], target_precision=tp) if len(unresolved) > 20 else thr[f"jev@{tag}"]
print("thresholds", thr)
json.dump({"thresholds": thr, "frozen_at": datetime.now(timezone.utc).isoformat()}, open(os.path.join(OUT, "thresholds.json"), "w"), indent=1)

# --- held-out evaluation ---------------------------------------------------------------
test["sup_score"] = supervised_scores(clf, test)
decisions = {"A_exact": exact_decision(test), "rules_only": test["rule_decision"]}
for tag in TARGETS:
    decisions[f"B_fuzzy_rules@{tag}"] = fuzzy_with_rules(test, *thr[f"fuzzy@{tag}"])
    decisions[f"C_supervised@{tag}"] = threshold_decision(test["sup_score"], *thr[f"supervised@{tag}"])
    if jev_ran:
        decisions[f"D_jev_pool@{tag}"] = jev_decision(test, *thr[f"jev@{tag}"])
        decisions[f"E_hybrid@{tag}"] = hybrid_decision(test, *thr[f"hybrid@{tag}"])

summary = []
for name, dec in decisions.items():
    m = pair_metrics(test, dec, n_gold_test)
    ci = grouped_bootstrap_ci(test, dec, n_gold_test, n_boot=300)
    m.update({f"{k}_ci": f"[{lo:.3f}, {hi:.3f}]" for k, (lo, hi) in ci.items()})
    m["method"] = name
    summary.append(m)
    test[f"dec_{name}"] = dec.values
    eb = error_breakdown(test, dec)
    eb.to_csv(os.path.join(OUT, f"errors_{name}.csv"), index=False)
summary_df = pd.DataFrame(summary).set_index("method")
print(summary_df[["precision", "recall_e2e", "f1_e2e", "review_fraction", "auto_coverage", "auto_accuracy", "fp", "fn_e2e"]].round(3))
summary_df.to_csv(os.path.join(OUT, "summary_test.csv"))

# --- matched-review-burden / matched-precision curves (score-based methods) --------------
curves = []
for name, score in (("B_fuzzy", test["fuzzy_score"]), ("C_supervised", test["sup_score"]),
                    ("D_jev", test["jev_score"] if jev_ran else None)):
    if score is None:
        continue
    sub = test[score.notna()]
    sc = score[score.notna()]
    for hi in np.linspace(0.3, 1.0, 36):
        dec = threshold_decision(sc, hi, hi)  # no review band: pure precision/recall trade-off
        m = pair_metrics(sub, dec, n_gold_test)
        curves.append({"method": name, "threshold": hi, **{k: m[k] for k in ("precision", "recall_e2e", "fp", "tp")}})
pd.DataFrame(curves).to_csv(os.path.join(OUT, "pr_curves.csv"), index=False)

# review burden at matched precision: for each method, min review fraction achieving >=0.99 precision on test (post-hoc, reported as such)
burden = []
for name, score in (("B_fuzzy", test["fuzzy_score"]), ("C_supervised", test["sup_score"]), ("D_jev", test["jev_score"] if jev_ran else None)):
    if score is None:
        continue
    sub = test[score.notna()]
    sc = score[score.notna()]
    for target in (0.95, 0.99):
        hi, lo = choose_thresholds(sc, sub["label"], target_precision=target)
        m = pair_metrics(sub, threshold_decision(sc, hi, lo), n_gold_test)
        burden.append({"method": name, "target_precision": target, "hi": hi, "lo": lo, **m})
pd.DataFrame(burden).to_csv(os.path.join(OUT, "matched_precision_posthoc.csv"), index=False)

test.to_csv(os.path.join(OUT, "test_pairs_scored.csv"), index=False)
dev.to_csv(os.path.join(OUT, "dev_pairs_scored.csv"), index=False)

# --- plots ------------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cdf = pd.DataFrame(curves)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, g in cdf.groupby("method"):
        ax.plot(g["recall_e2e"], g["precision"], marker=".", label=name, color={"B_fuzzy": "#999", "C_supervised": "#555", "D_jev": "#000"}[name])
    ax.set_xlabel("end-to-end recall (incl. retrieval misses)"); ax.set_ylabel("precision"); ax.set_title(f"{args.benchmark}: test precision/recall")
    ax.axhline(0.99, ls="--", color="#bbb"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pr_curve.png"), dpi=150)
    fig, ax = plt.subplots(figsize=(7, 4))
    s = summary_df[["precision", "recall_e2e", "review_fraction"]]
    s.plot.bar(ax=ax, color=["#000", "#777", "#ccc"]); ax.set_ylim(0, 1); ax.set_title("held-out test, frozen thresholds"); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "summary_bars.png"), dpi=150)
except Exception as e:  # plotting must never block results
    print("plot failed", e)

meta = {"benchmark": args.benchmark, "k": args.k, "jev_topk": args.jev_topk, "dev_left_for_jev": args.dev_left,
        "split": manifest, "retrieval": retr, "jev": jev_status, "thresholds": thr,
        "runtime_s": round(time.time() - t0, 1), "finished_at": datetime.now(timezone.utc).isoformat()}
json.dump(meta, open(os.path.join(OUT, "run_meta.json"), "w"), indent=1, default=str)
print("done in", meta["runtime_s"], "s")
