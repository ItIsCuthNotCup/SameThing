# SameThing — Results

Research question: **Can Jev reduce manual product-matching work while maintaining better precision than conventional fuzzy matching?**

Short answer on Abt-Buy (held-out, group-aware split): **partly yes.** Adding Jev to the same rules + candidate pool raised end-to-end recall from 0.53 to 0.78 at essentially the same observed precision (0.969 → 0.977) and cut the review queue from 1,710 to 1,573 pairs, while introducing one new false match. **Nobody, including Jev, reached the 99% precision target with a confidence interval that clears it.** Jev's own pool-only judgments were the most precise method (0.994, CI 0.986–1.000) but the hybrid inherits the deterministic rules' false matches.

Everything below is from `results/abt-buy/` (`summary_test.csv`, `run_meta.json`, `thresholds.json`, `errors_*.csv`, `pr_curve.png`, `summary_bars.png`).

## Data and provenance

| | |
|---|---|
| Benchmark | Abt-Buy, Database Group, University of Leipzig (Köpcke, Thor, Rahm, VLDB 2010) |
| Page | https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution |
| File | https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip — sha256 `b9ff2937d97371b7a00b0a97033213bfbdee24e8f1a0718a41ea8364472a8ff8` |
| Downloaded | 2026-09-20 (UTC), see `data/raw/PROVENANCE.json` |
| Size | Abt 1,081 records, Buy 1,092 records, 1,097 gold pairs (`abt_buy_perfectMapping.csv`) |
| Match definition | The benchmark's mapping links records that describe the same product; it is used as-is for **product identity**. It does not label quantity/condition/bundle differences, so **offer comparability is unvalidated** here — the tool flags it but no metric claims accuracy for it. |
| Limitations | Historical (c. 2010) catalogs; prices are not current offers; Buy has sparse descriptions; the mapping is not guaranteed one-to-one. The dataset is public and may have appeared in Jev's training data. |

Amazon-GoogleProducts was downloaded (sha256 in `PROVENANCE.json`) and is loadable in the app, but was **not evaluated** (time).

## Split (no leakage)

Connected components over gold links were assigned whole to dev or test (seed 13, 40% dev). Unmatched records are singleton groups and are kept.

| split | Abt | Buy | gold pairs |
|---|---|---|---|
| dev | 433 | 435 | 438 |
| test | 648 | 657 | 659 |

Manifest: `results/abt-buy/split_manifest.csv`. Thresholds and the prompt were chosen on dev only and frozen (`thresholds.json`, `frozen_at` 2026-09-20T19:48Z) before the test metrics were computed. One prompt variant (`v1`) was used; the exact six Choice questions are in `samething/jev.py` (`QUESTION_SETS`) and shown in the app.

Jev was run on a **dev subsample**: 120 randomly chosen dev Abt records (seed 7) × top-3 candidates = 480 calls, used to pick the Jev/hybrid thresholds. On test, Jev saw **all** top-3 candidates of every test Abt record (1,944 pairs). Test numbers are full-test-split numbers, not a subset.

## Pipeline (same visible information for every method)

1. Normalize: case/punctuation/whitespace, model-token extraction with spec-token exclusions (`1080p`, `24port`, …), catalog-frequency pruning (tokens in >8 records are not identifiers), and stripping of compatibility clauses ("handset for the KX-TG6700B" → the KX token is a *compatibility* reference, not this product's model). Original values are kept.
2. Retrieve: union of exact model-token index, brand+model blocking, word TF-IDF, char 3–5-gram TF-IDF, top-8 per Abt record. Gold mapping never used.
   - **Candidate recall@8 = 0.997** on test (657/659; misses listed in `run_meta.json`). **Jev pool recall@3 = 0.961** (633/659): 26 gold pairs never reach Jev and are counted as retrieval failures in end-to-end recall.
3. Rules: identifier agreement + similar title → match; model conflict / capacity conflict / brand mismatch without identifier / compatibility reference / accessory-vs-product → different; variant suffix, qty or condition conflict → review.
4. Baselines: A exact normalized title/model; B rules + RapidFuzz composite score (thresholds from dev); C logistic regression on 25 pair features (trained on dev).
5. Jev (`jev-1.13.0`, TypeSafe `/v1/systemone`): six independent Choice questions per pair; score = P(same) on the verdict question, then explicit conflict answers (identifier conflict, variant conflict, accessory) veto a match in code. Failed calls become review — there were 0 failures.
6. Hybrid E: rule decision if the rules resolve the pair, else Jev score vs dev thresholds; qty/condition conflicts are never auto-matched.

## Held-out results (test split, 7,207 candidate pairs, 659 gold)

Thresholds tuned on dev for a **99% precision target** (and 95% in the second block). CIs are 95% bootstrap over Abt records (approximates grouping by product).

| method | precision (CI) | recall e2e (CI) | F1 | FP | missed | review fraction | auto coverage | auto accuracy |
|---|---|---|---|---|---|---|---|---|
| A exact | 0.980 (0.962–0.993) | 0.608 (0.570–0.644) | 0.751 | 8 | 258 | 0.000 | 1.000 | 0.963 |
| rules only | 0.969 (0.942–0.991) | 0.524 | 0.680 | 11 | 314 | 0.242 | 0.758 | 0.988 |
| **B fuzzy+rules @99** | 0.969 (0.943–0.991) | 0.528 (0.491–0.562) | 0.684 | 11 | 311 | 0.237 | 0.763 | 0.988 |
| C supervised @99 | 0.959 (0.901–1.000) | 0.106 | 0.191 | 3 | 589 | 0.544 | 0.456 | 0.998 |
| D Jev pool @99 | **0.994 (0.986–1.000)** | 0.780 (0.752–0.808) | 0.874 | 3 | 145 | n/a¹ | n/a¹ | 0.994 |
| **E hybrid @99** | 0.977 (0.960–0.992) | **0.783 (0.754–0.813)** | **0.869** | 12 | 143 | **0.218** | 0.782 | 0.989 |
| B fuzzy+rules @95 | 0.969 | 0.528 | 0.684 | 11 | 311 | 0.237 | 0.763 | 0.988 |
| C supervised @95 | 0.967 | 0.445 | 0.609 | 10 | 366 | 0.512 | 0.488 | 0.996 |
| D Jev pool @95 | 0.982 | 0.825 | 0.897 | 10 | 115 | n/a¹ | n/a¹ | 0.982 |
| E hybrid @95 | 0.976 | 0.801 | 0.880 | 13 | 131 | 0.216 | 0.784 | 0.988 |

¹ D only judges the 1,944 top-3 pairs; the other 5,263 candidate pairs are counted as "review" in `summary_test.csv` (0.928), which is an artifact of the pool, not a real queue. Within the pool, Jev's raw verdict was: same → 558 TP / 4 FP; different → 1,271 TN / 50 FN; insufficient → 61 pairs.

"Missed" = gold pairs not auto-matched, including the 2 not retrieved at k=8 and (for D/E) the 26 not in the top-3 pool.

Note that "A exact" looks strong here because the Abt title suffix is almost always a model number and Buy often repeats it; it has no review queue at all and simply misses 39% of pairs.

### Post-hoc "what threshold would have hit 99%?"
`matched_precision_posthoc.csv` sweeps the *test* threshold for B (this is peeking, reported only as a diagnostic): B cannot reach 0.99 precision at any threshold without recall collapsing below 0.18. The dev-selected 99% threshold for B in fact produced 0.969 on test — the dev estimate was optimistic.

## Answers

**1. Did Jev improve matching over the strongest baseline?**
Yes on recall/F1 at comparable precision: E hybrid 0.783 recall vs B 0.528 (CIs do not overlap) at 0.977 vs 0.969 precision (CIs overlap). Jev alone on the pool (D) was the most precise method observed (3 FP in 517 matches). The strongest conventional baseline is B (C, the supervised model, overfit to dev at these targets and had worse recall at every operating point).

**2. How much manual review did it remove at comparable precision?**
Review queue 1,710 → 1,573 candidate pairs (−137, −8% of the queue; review fraction 0.237 → 0.218) at ±1 FP. More importantly, 168 true matches that B left for humans were auto-accepted. The remaining hybrid queue is dominated by pairs the rules could not resolve and which were *outside* Jev's top-3 pool (Jev was only allowed to see 3 candidates per record to fit the budget; with the full k=8 pool the queue would shrink further at ~2.5× the cost, ≈$0.35).
*Labor estimate (assumption, not measured):* at 20 s per reviewed pair, 137 fewer pairs ≈ 46 minutes saved per 650 matched products, plus 168 correct matches that would otherwise have needed a human decision (≈ 56 min). Cost of the Jev calls that produced this: ≈ $0.10.

**3. Which mistakes did it prevent?**
168 missed matches from B became correct matches, mostly "identifier missing / wording" cases: `Sony Switcher - SBV40S` ↔ `Sony SB-V40S A/V Selector`; `Panasonic 2-Line Integrated Telephone - KXTSC14W` ↔ `Panasonic KX-TSC14W 2-Line Telephone`; `Omnimount Wall Speaker Mount - 20WLWH` ↔ `OmniMount Stainless Steel Speaker Mount - 20.0 WALL-W`. Missed "quantity/multipack" errors fell 48 → 7 and "identifier missing" 108 → 59 (see `errors_*.csv`). Jev did **not** overturn any rule decision (rules run first), so it prevented none of the rule false matches.

**4. Which mistakes did it introduce?**
One new false match: `Panasonic VIERA … TH50PZ85U` ↔ `Panasonic Viera TH-50PZ850U` (P(same)=0.56, just above the 0.509 threshold; the gold says these are different models). In the pool it also has 50 false "different" verdicts, which in the hybrid become review rather than auto-rejects (the dev-selected lower threshold was 0.0, i.e. Jev is never trusted to auto-reject). At the 95% target it adds 2 more FPs (`Canon CLI8PM` magenta *photo* ink vs `CLI-8M` magenta ink is a representative variant miss).

**5. Were the largest failures in retrieval or pair judgment?**
Pair judgment. Retrieval failed on 2/659 at k=8 and 26/659 at the k=3 Jev pool. The remaining 117 hybrid misses are pairs in the pool that Jev scored below threshold or left as insufficient (78) plus rule-level review. The 11 shared false matches all come from the deterministic identifier rule: colour variants sharing a base model (`DVP-FX820` black/blue/red ↔ `DVPFX820`, `YPS2ZW` ↔ `YP-S2ZG`), grill SKUs (`S-310` LP vs NG), and an aperture string `f/4-5.6` that survived pruning. These are the first things to fix.

**6. What did the experiment cost?**
2,419 API attempts (475 dev, 1,944 test), 0 failures, 0 retries needed; 3.01 M input tokens, 0.61 M output tokens; **≈ $0.126** at the documented $0.042 / M input tokens (output free); p50 latency 0.13 s, p95 0.24 s, concurrency 4; whole eval ran in 100 s. Budget limits were $3 / 3,000 attempts; neither was approached. Cached responses (no secrets) are in `data/cache/jev_cache.sqlite` (2,398 entries).

**7. What remains before someone should use it on a real catalog?**
- The 99% precision target is **not demonstrated**: best hybrid CI is 0.960–0.992. Fix the identifier rule (require suffix agreement, treat colour/region suffixes as variants), then re-tune.
- Offer comparability (qty/condition/bundle) is flagged but **unvalidated** — Abt-Buy has no labels for it.
- Everything was tuned on one 2010 electronics benchmark that may be in Jev's training data; performance on a new catalog is unproven. Run on a held-out sample of the real catalog with human labels first.
- Jev only saw 3 candidates per record; a real run should send all unresolved pairs (and consider both directions, Buy→Abt).
- Matching is not forced one-to-one (the benchmark isn't); a real reconciliation may need a post-hoc assignment step.
- Currency is assumed USD for the demo; price comparisons are labeled historical and shown only when currency/qty/condition are comparable, but shipping/tax are unknown.
- No auth/multi-user state in the Streamlit demo; review decisions live in the session and exports.

## Representative examples

Successes (hybrid): identifier-agreement rule — `Panasonic Corded Phone - KXTS3282B` ↔ `Panasonic KX-TS3282B Corded Phone`; Jev on wording — `Escort Passport Radar And Laser Detector - Black Finish - 8500` ↔ `Escort X50 Passport 8500 Radar Detector … 8500X50RED` (P=0.77, gold match); compatibility rule — `Add-On Handset For The KXTG6700B` correctly kept apart from the KX-TG6700B phone.

Failures: rule FP on colour variants (`DVP-FX820` red/blue vs plain); Jev FP `TH50PZ85U` vs `TH-50PZ850U`; Jev "insufficient" on `Bose Acoustimass 5 Series III - AM53BK` ↔ same title with `- 21725` (gold match, sent to review — arguably correct behaviour given the conflicting identifiers).

## Caveats

- Public benchmark; possible training-data contamination for Jev.
- Confidence values returned by Jev were stored (`jev_*_conf`) but are a model signal, not measured accuracy; they were not used in decisions.
- Precision CIs are bootstrapped by Abt record, which approximates but is not exactly grouping by underlying product.
- Prices are historical; differences shown in the app are not available savings.
