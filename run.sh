#!/usr/bin/env bash
# One-command launch: creates a venv, installs deps, fetches the public benchmark, starts the app.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python scripts/fetch_data.py
.venv/bin/streamlit run app.py
