"""Candidate retrieval: union of cheap blocking methods, no reference labels used."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import Record, tokens


def _topk_from_matrix(sim, k):
    out = {}
    for i in range(sim.shape[0]):
        row = sim[i]
        if hasattr(row, "toarray"):
            row = row.toarray().ravel()
        idx = np.argpartition(-row, min(k, len(row) - 1))[:k]
        out[i] = [(int(j), float(row[j])) for j in idx if row[j] > 0]
    return out


def retrieve_candidates(left: list[Record], right: list[Record], k: int = 8) -> dict[str, list[dict]]:
    """Return {left_rid: [{right_rid, methods, tfidf_char, tfidf_tok}]} with at most ~k*2 candidates.

    Methods:
      exact_model   - shared canonical model token (explicit model field or extracted)
      brand_model   - same brand and a shared model token of length>=5
      token_tfidf   - word TF-IDF cosine top-k
      char_tfidf    - char 3-5gram TF-IDF cosine top-k
    """
    cands: dict[str, dict[str, dict]] = defaultdict(dict)

    def add(li, rj, method, score=None):
        lr, rr = left[li].rid, right[rj].rid
        c = cands[lr].setdefault(rr, {"right_rid": rr, "methods": set(), "tfidf_char": 0.0, "tfidf_tok": 0.0})
        c["methods"].add(method)
        if score is not None:
            c[method] = max(c.get(method, 0.0), score)

    # 1. model token index
    idx = defaultdict(list)
    for j, r in enumerate(right):
        for t in set(r.model_tokens) | set(r.desc_model_tokens) | set(r.compat_model_tokens):
            idx[t].append(j)
    for i, l in enumerate(left):
        for t in set(l.model_tokens) | set(l.desc_model_tokens) | set(l.compat_model_tokens):
            for j in idx.get(t, []):
                if len(idx[t]) > 25:
                    continue
                add(i, j, "exact_model")
                if len(t) >= 5 and l.brand_norm and l.brand_norm == right[j].brand_norm:
                    add(i, j, "brand_model")

    ltext = [f"{r.title_norm} {r.desc_norm[:200]}" for r in left]
    rtext = [f"{r.title_norm} {r.desc_norm[:200]}" for r in right]
    ltitle = [r.title_norm for r in left]
    rtitle = [r.title_norm for r in right]

    # 2. word tfidf on title+desc snippet
    vec = TfidfVectorizer(tokenizer=tokens, token_pattern=None, lowercase=False, sublinear_tf=True)
    vec.fit(ltext + rtext)
    sim = (vec.transform(ltext) @ vec.transform(rtext).T)
    for i, lst in _topk_from_matrix(sim, k).items():
        for j, s in lst:
            add(i, j, "token_tfidf", s)
            cands[left[i].rid][right[j].rid]["tfidf_tok"] = s

    # 3. char n-gram tfidf on titles
    cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    cvec.fit(ltitle + rtitle)
    csim = (cvec.transform(ltitle) @ cvec.transform(rtitle).T)
    for i, lst in _topk_from_matrix(csim, k).items():
        for j, s in lst:
            add(i, j, "char_tfidf", s)
            cands[left[i].rid][right[j].rid]["tfidf_char"] = s

    # fill in similarity scores for candidates found only by identifier methods
    rpos = {r.rid: j for j, r in enumerate(right)}
    Lc, Rc = cvec.transform(ltitle), cvec.transform(rtitle)
    Lt, Rt = vec.transform(ltext), vec.transform(rtext)
    out = {}
    for i, l in enumerate(left):
        lst = list(cands[l.rid].values())
        for c in lst:
            j = rpos[c["right_rid"]]
            if c["tfidf_char"] == 0.0:
                c["tfidf_char"] = float((Lc[i] @ Rc[j].T).toarray()[0, 0])
            if c["tfidf_tok"] == 0.0:
                c["tfidf_tok"] = float((Lt[i] @ Rt[j].T).toarray()[0, 0])
            c["methods"] = sorted(c["methods"])
        lst.sort(key=lambda c: -(c["tfidf_char"] + c["tfidf_tok"] + (0.5 if "exact_model" in c["methods"] else 0)))
        out[l.rid] = lst[: 2 * k]
    return out


def candidate_recall(cands: dict[str, list[dict]], gold_pairs: set[tuple[str, str]]) -> dict:
    found = 0
    misses = []
    for l, r in gold_pairs:
        if any(c["right_rid"] == r for c in cands.get(l, [])):
            found += 1
        else:
            misses.append((l, r))
    n_pairs = sum(len(v) for v in cands.values())
    return {"gold": len(gold_pairs), "found": found, "recall": found / max(1, len(gold_pairs)),
            "n_candidate_pairs": n_pairs, "misses": misses}
