# SameThing

Matches equivalent products across two messy catalogs: proposes matches, flags material differences (model variant, capacity, quantity, condition, bundle, accessory-vs-product), and leaves uncertain pairs for review. Source rows are never modified; the output is a mapping plus review decisions.

Results of the Abt-Buy evaluation, including whether Jev helps: **[RESULTS.md](RESULTS.md)**.

## Run

```bash
./run.sh          # creates .venv, installs deps, downloads the public benchmarks, starts the app
```

Then open http://localhost:8501. To enable Jev-assisted matching set `TYPESAFE_API_KEY` in the server's environment; the key never reaches the browser. Without it the app runs conventional matching only.

Manual equivalent: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python scripts/fetch_data.py && .venv/bin/streamlit run app.py`.

## Reproduce the evaluation

```bash
.venv/bin/python scripts/run_eval.py --out results/abt-buy            # uses cached Jev responses in data/cache/
.venv/bin/python scripts/run_eval.py --no-jev --out results/nojev     # conventional baselines only
```

Budget guard defaults: 2,700 attempts / $3. Outputs: `summary_test.csv`, `errors_<method>.csv`, `pr_curve.png`, `summary_bars.png`, `split_manifest.csv`, `thresholds.json`, `run_meta.json`, scored pair CSVs.

## Layout

- `samething/normalize.py` — column mapping, source-preserving normalization, model/variant/qty/condition extraction
- `samething/candidates.py` — candidate retrieval (identifier index, brand+model blocking, word and char n-gram TF-IDF)
- `samething/features.py` — pair features, deterministic rules, material-difference detection
- `samething/jev.py` — TypeSafe Jev client: six Choice questions, cache, budget, retries
- `samething/pipeline.py` — baselines, supervised matcher, hybrid decision, explanations
- `samething/evaluate.py` — metrics, grouped bootstrap CIs, threshold selection, error taxonomy
- `scripts/run_eval.py`, `scripts/fetch_data.py`
- `app.py` — Streamlit interface
- `data/raw/PROVENANCE.json` — dataset URLs, checksums, download dates
- `docs/screenshots/` — interface screenshots
