"""SameThing - match equivalent products across messy catalogs. Run: streamlit run app.py"""
from __future__ import annotations

import io
import json
import os
import zipfile

import pandas as pd
import streamlit as st

from samething import data as D
from samething.features import DIFFERENT, MATCH, REVIEW
from samething.jev import Budget, JevClient, QUESTION_SETS
from samething.normalize import ColumnMap, build_records
from samething.pipeline import (build_pairs, explain_pair, fuzzy_with_rules, hybrid_decision, run_jev)

st.set_page_config(page_title="SameThing", layout="wide")
st.markdown(
    """<style>
    :root { --gray: #666; }
    .stApp { background: #fff; color: #111; }
    h1, h2, h3 { color: #000; font-weight: 600; letter-spacing: -0.01em; }
    .stButton>button { background:#000; color:#fff; border:1px solid #000; border-radius:4px; }
    .stButton>button:hover { background:#333; color:#fff; border-color:#333; }
    .stDownloadButton>button { background:#fff; color:#000; border:1px solid #000; border-radius:4px; }
    section[data-testid="stSidebar"] { background:#f4f4f4; border-right:1px solid #ddd; }
    .tag { display:inline-block; padding:2px 8px; border:1px solid #000; border-radius:3px; font-size:12px; margin-right:4px; }
    .tag.match { background:#000; color:#fff; } .tag.review { background:#fff; color:#000; } .tag.different { background:#ddd; color:#000; border-color:#ddd; }
    .evidence li { color:#333; font-size:14px; }
    .hist { color:#666; font-size:13px; }
    </style>""",
    unsafe_allow_html=True,
)

THRESH_PATH = os.path.join(D.ROOT, "results", "abt-buy", "thresholds.json")
DEFAULT_THR = {"fuzzy@99": (1.0, 0.1), "hybrid@99": (0.9, 0.3)}
if os.path.exists(THRESH_PATH):
    DEFAULT_THR.update(json.load(open(THRESH_PATH))["thresholds"])

ss = st.session_state
ss.setdefault("decisions", {})  # (left_rid,right_rid) -> accepted/rejected

st.title("SameThing")
st.caption("Match equivalent products across two catalogs. Proposes matches, flags material differences, leaves uncertain pairs for review. Source rows are never modified.")

# ---------------------------------------------------------------- sidebar: data + mapping
with st.sidebar:
    st.header("1. Catalogs")
    mode = st.radio("Source", ["Public demo: Abt-Buy", "Public demo: Amazon-Google", "Upload two CSVs"], index=0)
    left_df = right_df = None
    lmap = rmap = None
    if mode.startswith("Public"):
        bname = "abt-buy" if "Abt" in mode else "amazon-google"
        b = D.BENCHMARKS[bname]
        left_df, right_df = D.read_csv(b["left"]), D.read_csv(b["right"])
        lmap, rmap = b["left_map"], b["right_map"]
        lname, rname = b["left_name"], b["right_name"]
        st.caption(f"{lname}: {len(left_df)} rows, {rname}: {len(right_df)} rows. Historical 2010 data (Leipzig DBS benchmark).")
        limit = st.slider("Rows per catalog (demo speed)", 100, max(len(left_df), len(right_df)), 400, step=100)
        left_df, right_df = left_df.head(limit), right_df.head(limit)
    else:
        lf = st.file_uploader("Catalog A (CSV)", type="csv")
        rf = st.file_uploader("Catalog B (CSV)", type="csv")
        lname, rname = "A", "B"
        if lf and rf:
            left_df, right_df = pd.read_csv(lf, dtype=str, keep_default_na=False), pd.read_csv(rf, dtype=str, keep_default_na=False)

    def mapper(df, label):
        st.subheader(f"Columns: {label}")
        cols = ["<none>"] + list(df.columns)
        def pick(name, default):
            i = cols.index(default) if default in cols else 0
            v = st.selectbox(f"{label} {name}", cols, index=i, key=f"{label}_{name}")
            return None if v == "<none>" else v
        guess = lambda *names: next((c for c in df.columns if c.lower() in names), "<none>")
        return ColumnMap(id=pick("record id", guess("id")) or df.columns[0], title=pick("title", guess("name", "title")) or df.columns[1],
                         description=pick("description", guess("description")), brand=pick("brand", guess("manufacturer", "brand")),
                         model=pick("model number", guess("model", "mpn")), price=pick("price", guess("price")),
                         default_currency=st.text_input(f"{label} currency (if not a column)", "USD", key=f"{label}_cur") or None)

    if left_df is not None and mode.startswith("Upload"):
        lmap, rmap = mapper(left_df, "A"), mapper(right_df, "B")

    st.header("2. Run")
    k = st.slider("Candidates per record", 3, 10, 5)
    use_jev = st.checkbox("Also ask Jev on unresolved pairs (uses API budget)", value=False)
    jev_cap = st.number_input("Max Jev calls this run", 10, 2000, 200, step=10)
    run = st.button("Run matching", type="primary", disabled=left_df is None)
    api_ok = bool(os.environ.get("TYPESAFE_API_KEY") or os.environ.get("Jev"))
    st.caption(("Jev key found on server." if api_ok else "No TYPESAFE_API_KEY on server; Jev disabled.") + " Credentials never leave the server.")

if run:
    with st.spinner("Normalizing, retrieving candidates, scoring pairs..."):
        left = build_records(left_df, lmap, lname)
        right = build_records(right_df, rmap, rname)
        pairs, _ = build_pairs(left, right, k=k)
        hi, lo = DEFAULT_THR["fuzzy@99"]
        pairs["decision_conventional"] = fuzzy_with_rules(pairs, hi, lo)
        pairs["jev_score"] = float("nan")
        pairs["decision"] = pairs["decision_conventional"]
        jev_summary = None
        if use_jev and api_ok:
            client = JevClient(os.path.join(D.ROOT, "data", "cache", "jev_cache.sqlite"), budget=Budget(max_attempts=int(jev_cap)))
            mask = (pairs["rule_decision"] == REVIEW)
            mask &= mask.cumsum() <= int(jev_cap)
            pairs = run_jev(pairs, left, right, client, mask)
            hi, lo = DEFAULT_THR.get("hybrid@99", DEFAULT_THR["fuzzy@99"])
            pairs["decision"] = hybrid_decision(pairs, hi, lo)
            jev_summary = client.budget.summary()
        ss["pairs"], ss["left"], ss["right"], ss["jev_summary"] = pairs, {r.rid: r for r in left}, {r.rid: r for r in right}, jev_summary
        ss["decisions"] = {}

if "pairs" not in ss:
    st.info("Choose catalogs in the sidebar and press **Run matching**.")
    with st.expander("Jev questions used (exact text)"):
        st.json(QUESTION_SETS["v1"])
    st.stop()

pairs: pd.DataFrame = ss["pairs"]
L, R = ss["left"], ss["right"]

# ---------------------------------------------------------------- summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidate pairs", len(pairs))
c2.metric("Proposed matches", int((pairs.decision == MATCH).sum()))
c3.metric("Needs review", int((pairs.decision == REVIEW).sum()))
c4.metric("Different", int((pairs.decision == DIFFERENT).sum()))
if ss.get("jev_summary"):
    b = ss["jev_summary"]
    st.caption(f"Jev: {b['attempts']} calls, {b['input_tokens']} input tokens, est. ${b['estimated_usd']:.4f}, {b['failures']} failures, p50 latency {b['latency_p50_s'] or 0:.2f}s. Model: {pairs['jev_model'].dropna().iloc[0] if 'jev_model' in pairs and pairs['jev_model'].notna().any() else 'n/a'}")

# ---------------------------------------------------------------- filter + table
f1, f2, f3 = st.columns([1, 1, 2])
show = f1.multiselect("Show", [MATCH, REVIEW, DIFFERENT], default=[MATCH, REVIEW])
only_diffs = f2.checkbox("Only pairs with material differences")
q = f3.text_input("Search title")
view = pairs[pairs.decision.isin(show)].copy()
if only_diffs:
    view = view[view.material_diffs.astype(str).str.len() > 0]
view["left_title"] = view.left_rid.map(lambda i: L[i].title)
view["right_title"] = view.right_rid.map(lambda i: R[i].title)
if q:
    view = view[view.left_title.str.contains(q, case=False) | view.right_title.str.contains(q, case=False)]
view["human"] = [ss["decisions"].get((a, b), "") for a, b in zip(view.left_rid, view.right_rid)]
view = view.sort_values(["decision", "fuzzy_score"], ascending=[True, False])
st.dataframe(view[["decision", "human", "left_title", "right_title", "fuzzy_score", "jev_score", "material_diffs", "rule_reason"]].round(3),
             use_container_width=True, height=280, hide_index=True)

# ---------------------------------------------------------------- pair inspector
st.subheader("Inspect a pair")
if len(view) == 0:
    st.write("No pairs in this filter.")
    st.stop()
labels = [f"{r.decision.upper()} | {r.left_title[:50]}  <->  {r.right_title[:50]}" for r in view.itertuples()]
sel = st.selectbox("Pair", range(len(labels)), format_func=lambda i: labels[i])
row = view.iloc[sel]
a, b = L[row.left_rid], R[row.right_rid]
cls = row.decision
st.markdown(f'<span class="tag {cls}">{cls}</span> <span class="tag">conventional: {row.decision_conventional}</span> ' +
            (f'<span class="tag">human: {ss["decisions"][(a.rid, b.rid)]}</span>' if (a.rid, b.rid) in ss["decisions"] else ""), unsafe_allow_html=True)
ca, cb = st.columns(2)
for col, rec, name in ((ca, a, a.source), (cb, b, b.source)):
    with col:
        st.markdown(f"**{name}** · id `{rec.rid}`")
        st.write(rec.title)
        st.caption(rec.description[:500] or "(no description)")
        st.text(f"brand: {rec.brand or rec.brand_norm or '?'}   model tokens: {', '.join(rec.model_tokens) or '-'}\n"
                f"price: {rec.price_raw or 'unknown'}   qty: {rec.pack_qty or 1}   condition: {rec.condition}")

st.markdown("**Why these were linked / separated**")
st.markdown('<ul class="evidence">' + "".join(f"<li>{e}</li>" for e in explain_pair(row)) + "</ul>", unsafe_allow_html=True)

# price comparison only when comparable
if a.price and b.price and a.currency == b.currency and (a.pack_qty or 1) == (b.pack_qty or 1) and not row.condition_conflict and row.decision == MATCH:
    diff = a.price - b.price
    st.markdown(f'<div class="hist">Historical price difference: {a.source} {a.price:.2f} vs {b.source} {b.price:.2f} {a.currency} '
                f'(difference {diff:+.2f}). These are historical benchmark prices, not available savings; shipping and tax treatment unknown.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="hist">Price difference not shown: prices missing, currency/quantity/condition not comparable, or pair not a proposed match.</div>', unsafe_allow_html=True)

b1, b2, b3 = st.columns([1, 1, 6])
if b1.button("Accept match"):
    ss["decisions"][(a.rid, b.rid)] = "accepted"; st.rerun()
if b2.button("Reject match"):
    ss["decisions"][(a.rid, b.rid)] = "rejected"; st.rerun()

# ---------------------------------------------------------------- export
st.subheader("Export")
def exports():
    p = pairs.copy()
    p["human_decision"] = [ss["decisions"].get((x, y), "") for x, y in zip(p.left_rid, p.right_rid)]
    p["left_title"] = p.left_rid.map(lambda i: L[i].title); p["right_title"] = p.right_rid.map(lambda i: R[i].title)
    final = p[(p.decision == MATCH) & (p.human_decision != "rejected") | (p.human_decision == "accepted")]
    mapping = final[["left_rid", "right_rid", "left_title", "right_title", "decision", "human_decision", "fuzzy_score", "jev_score", "material_diffs"]]
    matched_l = set(mapping.left_rid); matched_r = set(mapping.right_rid)
    unmatched = pd.DataFrame([{"source": r.source, "rid": r.rid, "title": r.title} for r in list(L.values()) + list(R.values())
                              if (r.rid not in matched_l and r.source == a.source) or (r.rid not in matched_r and r.source == b.source)])
    review = p[p.decision == REVIEW][["left_rid", "right_rid", "left_title", "right_title", "human_decision", "fuzzy_score", "jev_score", "material_diffs", "rule_reason"]]
    return mapping, unmatched, review
mapping, unmatched, review = exports()
e1, e2, e3, e4 = st.columns(4)
e1.download_button("mappings.csv", mapping.to_csv(index=False), "mappings.csv")
e2.download_button("unmatched.csv", unmatched.to_csv(index=False), "unmatched.csv")
e3.download_button("review_decisions.csv", review.to_csv(index=False), "review_decisions.csv")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("mappings.csv", mapping.to_csv(index=False)); z.writestr("unmatched.csv", unmatched.to_csv(index=False)); z.writestr("review_decisions.csv", review.to_csv(index=False))
e4.download_button("all (zip)", buf.getvalue(), "samething_export.zip")
