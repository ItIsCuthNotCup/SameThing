---
name: testing-samething
description: Exercise SameThing catalog matching, human review, CSV exports, and Jev through Streamlit.
---

# Runtime testing

Run from the repository root using the existing virtual environment:
`.venv/bin/streamlit run app.py --server.headless true --server.port 8501`.
Open http://localhost:8501. No app login is required.

## Devin Secrets Needed

- `TYPESAFE_API_KEY` (or `Jev` fallback) must be in the server environment for
  Jev-assisted testing. Verify the sidebar key-found indicator, never print keys.

## Focused flow

- Use Abt-Buy demo defaults (400 rows, 5 candidates), initially with Jev OFF.
- Run matching, remove match from Show, inspect a review pair, accept it,
  reject a distinct review pair, and download mappings/review_decisions CSVs.
- Verify actual downloaded files: accepted review pairs appear in mappings,
  rejected pairs do not, and review_decisions preserves both human_decision values.
- Download before running matching again: each run resets human decisions.
- Jev cache hits can yield fewer network calls than the cap. Assert the expected
  model and zero failures, rather than requiring the call count to equal the cap.
  Runtime Jev calls can modify the tracked data/cache/jev_cache.sqlite file.
- Original data/raw/abt-buy/Abt.csv and Buy.csv are Latin-1. Upload both to test
  encoding fallback, then inspect both mapping groups and run matching.
- Upload mode has an independently scrolling sidebar. Matching freshly uploaded
  files should show sources A/B; old results can remain visible until Run matching.

## Evidence

Streamlit main content and sidebar scroll independently. Chrome full-size
screenshot capture may only include their visible contents; take explicit
inspector/export screenshots as well. On macOS, use `cmd` shortcuts. If typing
a file path into the native picker fails, put the public path in the clipboard
with `pbcopy`, use Cmd+Shift+G and Cmd+V, then confirm the selection.
