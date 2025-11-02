#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NUCC Specialty Standardizer — offline, deterministic
----------------------------------------------------
Maps raw provider specialties to NUCC taxonomy codes using a hybrid approach:
1) Rule-based normalization + synonyms
2) Exact/normalized matching to NUCC fields
3) Character n-gram TF-IDF cosine similarity
4) Token Jaccard overlap
5) Fuzzy ratio via difflib.SequenceMatcher
6) Optional phonetic cue via simple Soundex

Combines scores to produce a confidence and returns top codes above a threshold;
emits "JUNK" when nothing clears the bar. No external API calls.

CLI:
python standardize.py --nucc nucc_taxonomy_master.csv --input input.csv --out output.csv   [--syn synonyms.csv] [--threshold 0.6] [--max-candidates 6]

Input columns:
- raw_specialty  (string)

Output columns:
- raw_specialty
- nucc_codes     (pipe-separated codes or JUNK)
- confidence     (0..1)
- explain        (short rationale)

Determinism:
- No randomness; ties broken lexicographically by (score desc, code asc)

Dependencies (soft, auto-fallback):
- pandas, numpy, scikit-learn (TF-IDF). If sklearn is missing, fall back to
  trigram Jaccard + difflib.
"""

import argparse
import difflib
import math
import re
import sys
from collections import defaultdict, Counter

# Soft deps
try:
    import pandas as pd
except Exception as e:
    print("ERROR: pandas is required.", file=sys.stderr)
    raise

try:
    import numpy as np
except Exception:
    np = None

# Optional scikit-learn; the script will degrade gracefully if unavailable
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Optional sentence-transformers embeddings (only if model is available locally)
EMB_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    import numpy as _np
    EMB_AVAILABLE = True
except Exception:
    EMB_AVAILABLE = False

NOISE_PATTERNS = [
    r"^dept(?:artment)?\s+of\s+",
    r"^department\s+of\s+",
    r"^clinic\s*[-–—:]?\s*",
    r"\bservices?\b",
    r"\bcenter\b",
    r"\bcentre\b",
    r"\bhospital\b",
    r"\bdivision\b",
    r"\bsection\b",
    r"\bunit\b",
    r"\bgroup\b",
    r"\bteam\b",
]

STOPWORDS = set("""
and the of for to in on with without adult pediatric pediatrician pediatricians
service services medicine medical health care general specialty speciality specialties
specialities doctor dr dept department hospital clinic center centre program practice
""".split())

SEPARATORS = r"[\/,;|\+\&]|(?:\band\b)"

# Optional: enrich stopwords with NLTK if available (no downloads performed here)
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    # If the corpus is already available, extend stopword list
    try:
        STOPWORDS |= set(w.lower() for w in nltk_stopwords.words("english"))
    except Exception:
        pass
except Exception:
    pass

# Optional: spaCy lemmatizer (if model is available locally)
_SPACY_NLP = None
try:
    import spacy
    # Try a lightweight default; if not available, skip silently
    try:
        _SPACY_NLP = spacy.load("en_core_web_sm", disable=["ner", "parser", "textcat"])
    except Exception:
        _SPACY_NLP = None
except Exception:
    _SPACY_NLP = None

def _lemmatize_tokens(tokens):
    if not _SPACY_NLP or not tokens:
        return tokens
    doc = _SPACY_NLP(" ".join(tokens))
    return [t.lemma_.lower() if t.lemma_ else t.text.lower() for t in doc]
PUNCT_TABLE = str.maketrans({c: " " for c in r"""!"#$%()*+,-./:;<=>?@[\]^_`{|}~"""})  # keep & handled separately

def normalize_text(s: str) -> str:
    """Lowercase, trim, remove noise words and punctuation, collapse spaces."""
    if not isinstance(s, str):
        return ""
    x = s.lower().strip()
    x = x.replace("&", " and ")
    # strip common noise prefixes/suffixes
    for pat in NOISE_PATTERNS:
        x = re.sub(pat, "", x)
    x = x.translate(PUNCT_TABLE)
    x = re.sub(r"\s+", " ", x).strip()
    # drop stopwords tokens only if string would remain non-empty
    toks = [t for t in x.split() if t not in STOPWORDS]
    return " ".join(toks) if toks else x

def tokenize(s: str):
    toks = [t for t in normalize_text(s).split() if t]
    toks = _lemmatize_tokens(toks)
    return toks

def char_ngrams(s: str, n=3):
    s = f" {normalize_text(s)} "
    return [s[i:i+n] for i in range(max(0, len(s)-n+1))]

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    i = len(a & b)
    u = len(a | b)
    return i / u if u else 0.0

def soundex(word: str) -> str:
    """Simple Soundex for crude phonetic similarity (english-biased)."""
    word = normalize_text(word)
    if not word:
        return ""
    first = word[0].upper()
    mapping = {
        **{c:"1" for c in "bfpv"},
        **{c:"2" for c in "cgjkqsxz"},
        **{c:"3" for c in "dt"},
        "l":"4",
        **{c:"5" for c in "mn"},
        "r":"6",
    }
    digits = []
    prev = ""
    for ch in word[1:]:
        d = mapping.get(ch, "")
        if d and d != prev:
            digits.append(d)
        prev = d
    code = (first + "".join(digits) + "000")[:4]
    return code

def seq_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def build_candidates(nucc_df):
    """
    Build a canonical candidate text per code:
    display_name (preferred) else "classification - specialization" else just classification.
    Also create a set of surfaces for matching and a normalized form.
    """
    records = []
    for _, row in nucc_df.iterrows():
        code = str(row.get("code", "")).strip()
        classification = str(row.get("classification", "") or "").strip()
        specialization = str(row.get("specialization", "") or "").strip()
        display = str(row.get("display_name", "") or "").strip()

        if display:
            cand = display
        elif classification and specialization:
            cand = f"{classification} - {specialization}"
        else:
            cand = classification or specialization

        norm = normalize_text(cand)
        surfaces = set(filter(None, {
            cand,
            classification,
            specialization,
            f"{classification} {specialization}".strip(),
            norm
        }))

        records.append({
            "code": code,
            "text": cand,
            "norm": norm,
            "surfaces": [normalize_text(s) for s in surfaces if s],
            "tokens": set(tokenize(cand)),
            "phon": set(soundex(t) for t in tokenize(cand))
        })
    return records

def load_synonyms(path):
    """
    synonyms.csv columns:
    alias, normalized, code (optional)
    - alias: what appears in raw input (e.g., "obgyn", "ent", "cardio")
    - normalized: canonical text to search among NUCC (e.g., "obstetrics and gynecology")
    - code: if provided, map directly to this NUCC code
    """
    syn = []
    try:
        import pandas as pd
        sdf = pd.read_csv(path)
        for _, r in sdf.iterrows():
            alias = str(r.get("alias", "")).strip().lower()
            normalized = str(r.get("normalized", "")).strip().lower()
            code_val = r.get("code", "")
            code = (None if (pd.isna(code_val) or str(code_val).strip()=="" or str(code_val).strip().lower()=="nan") else str(code_val).strip())
            syn.append((alias, normalized, code))
    except Exception:
        pass
    return syn

def build_tfidf(candidates):
    corpus = [c["norm"] for c in candidates]
    if SKLEARN_AVAILABLE:
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), lowercase=False, min_df=1)
        X = vec.fit_transform(corpus)
        return vec, X
    else:
        return None, None


def build_embeddings(candidates):
    """
    If sentence-transformers is available and a model is cached locally,
    precompute embeddings for candidate norms.
    The model name can be overridden via env var EMB_MODEL (default: all-MiniLM-L6-v2).
    """
    if not EMB_AVAILABLE:
        return None, None
    import os
    model_name = os.environ.get("EMB_MODEL", "all-MiniLM-L6-v2")
    try:
        model = SentenceTransformer(model_name)
        corpus = [c["norm"] for c in candidates]
        vecs = model.encode(corpus, normalize_embeddings=True, show_progress_bar=False)
        return model, vecs
    except Exception:
        return None, None

def embedding_scores(model, cand_vecs, query_norm):
    if not EMB_AVAILABLE or model is None or cand_vecs is None:
        return None
    q = model.encode([query_norm], normalize_embeddings=True, show_progress_bar=False)[0]
    import numpy as np
    sims = (cand_vecs @ q).astype(float)
    return sims


def tfidf_scores(vec, X, query_norm):
    if not SKLEARN_AVAILABLE or vec is None:
        return None
    import numpy as np
    q = vec.transform([query_norm])
    sims = cosine_similarity(q, X).reshape(-1)
    return sims

def trigram_jaccard_scores(candidates, query):
    """Fallback similarity based on trigram Jaccard."""
    q_tris = set(char_ngrams(query, 3))
    scores = []
    for i, c in enumerate(candidates):
        c_tris = set(char_ngrams(c["norm"], 3))
        scores.append(len(q_tris & c_tris) / (len(q_tris | c_tris) or 1))
    return scores

def choose_codes_for_phrase(phrase, candidates, tfidf_vec, tfidf_X, emb_model, emb_vecs, syn_index, threshold=0.6, k=8):
    """Return ([(code, score, how)], best_score_explain)"""
    raw = phrase
    norm = normalize_text(phrase)
    tokens = set(norm.split())
    phon = set(soundex(t) for t in tokens)

    # 0) Synonym direct mapping
    if norm in syn_index["alias->code"]:
        codes = syn_index["alias->code"][norm]
        out = [(c, 0.99, "synonym->code") for c in sorted(codes)]
        return out, f"synonym direct map for '{raw}'"

    # 1) Synonym normalized text (to be searched)
    if norm in syn_index["alias->normalized"]:
        norm = syn_index["alias->normalized"][norm]

    # 2) Exact normalized surface match
    matched_exact = []
    for c in candidates:
        if norm and norm in c["surfaces"]:
            matched_exact.append((c["code"], 0.98, "exact surface"))

    if matched_exact:
        # deduplicate
        uniq = {}
        for code, score, how in matched_exact:
            uniq[code] = max(uniq.get(code, 0), score)
        res = [(code, uniq[code], "exact surface") for code in sorted(uniq)]
        return res, f"exact match on '{norm}'"

    # 3) Similarity scoring
    if SKLEARN_AVAILABLE and tfidf_vec is not None:
        sims = tfidf_scores(tfidf_vec, tfidf_X, norm)
    else:
        sims = trigram_jaccard_scores(candidates, norm)

    # Optional embedding similarity
    emb_sims = embedding_scores(emb_model, emb_vecs, norm) if emb_model is not None else None

    # compute hybrid score per candidate
    scored = []
    for i, c in enumerate(candidates):
        tfidf = float(sims[i]) if sims is not None else 0.0
        emb = float(emb_sims[i]) if emb_sims is not None else 0.0
        jac = jaccard(tokens, c["tokens"])
        phon_overlap = jaccard(phon, c["phon"])
        fuzz = seq_ratio(norm, c["norm"])
        # Weighted hybrid; if embeddings present, include them
        if emb_sims is not None:
            hybrid = 0.4*tfidf + 0.2*emb + 0.25*fuzz + 0.1*jac + 0.05*phon_overlap
        else:
            hybrid = 0.5*tfidf + 0.25*fuzz + 0.2*jac + 0.05*phon_overlap
        scored.append((c["code"], hybrid, tfidf, fuzz, jac))

    # keep top-k unique by code
    by_code = {}
    for code, hybrid, tfidf, fuzz, jac in scored:
        if code not in by_code or hybrid > by_code[code][0]:
            by_code[code] = (hybrid, tfidf, fuzz, jac)

    ranked = sorted([(code, *vals) for code, vals in by_code.items()],
                    key=lambda x: (-x[1], x[0]))

    top = [(code, hybr, "hybrid") for code, hybr, _, _, _ in ranked[:k] if hybr >= threshold]
    if not top:
        # if nothing meets threshold, return the best one as a weak suggestion (but caller may mark JUNK)
        if ranked:
            code, hybr, tfidf, fuzz, jac = ranked[0]
            return [(code, hybr, "hybrid")], f"best below threshold (hybrid={hybr:.2f})"
        else:
            return [], "no candidates"
    else:
        return top, f"top {len(top)} above threshold"

def main():
    ap = argparse.ArgumentParser(description="NUCC specialty standardizer (offline)")
    ap.add_argument("--nucc", required=True, help="Path to nucc_taxonomy_master.csv")
    ap.add_argument("--input", required=True, help="Input CSV with column raw_specialty")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--syn", default=None, help="Optional synonyms.csv path")
    ap.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for accepting codes")
    ap.add_argument("--max-candidates", type=int, default=6, help="Max codes to emit per raw phrase (pipe-separated)")
    args = ap.parse_args()

    import pandas as pd

    nucc = pd.read_csv(args.nucc, dtype=str).fillna("")
    # Normalize column names to lowercase for robustness
    nucc.columns = [c.lower() for c in nucc.columns]
    candidates = build_candidates(nucc)

    # Build TF-IDF or fallback index
    tfidf_vec, tfidf_X = build_tfidf(candidates)
    # Optional embeddings
    emb_model, emb_vecs = build_embeddings(candidates)

    # Synonyms index
    from collections import defaultdict
    syn_index = {"alias->code": defaultdict(set), "alias->normalized": {}}
    if args.syn:
        syns = load_synonyms(args.syn)
        for alias, normalized, code in syns:
            if code:
                syn_index["alias->code"][alias].add(code)
            if normalized:
                syn_index["alias->normalized"][alias] = normalized

    # Load input
    df = pd.read_csv(args.input, dtype=str).fillna("")
    if "raw_specialty" not in df.columns:
        # try to locate plausible column
        for col in df.columns:
            if col.lower().strip() in {"raw_specialty", "raw", "specialty", "speciality"}:
                df = df.rename(columns={col: "raw_specialty"})
                break
    if "raw_specialty" not in df.columns:
        raise ValueError("Input CSV must contain a 'raw_specialty' column.")

    rows = []
    for raw in df["raw_specialty"].astype(str).tolist():
        if not raw.strip():
            rows.append((raw, "JUNK", 0.0, "empty"))
            continue

        parts = [p.strip() for p in re.split(SEPARATORS, raw, flags=re.IGNORECASE) if p.strip()]
        all_codes = []
        part_explains = []
        part_scores = []

        for p in parts:
            codes_scores, why = choose_codes_for_phrase(
                p, candidates, tfidf_vec, tfidf_X, emb_model, emb_vecs, syn_index,
                threshold=args.threshold, k=args.max_candidates
            )
            if codes_scores:
                all_codes.extend(codes_scores)
                part_scores.append(max(s for _, s, _ in codes_scores))
            else:
                part_scores.append(0.0)
            part_explains.append(f"[{p}: {why}]")

        # Aggregate across parts
        if not all_codes:
            rows.append((raw, "JUNK", 0.0, "; ".join(part_explains)))
            continue

        # Combine by code, keep max score
        by_code = {}
        for code, score, how in all_codes:
            by_code[code] = max(by_code.get(code, 0.0), float(score))

        # Keep top-N, stable sort by score desc then code asc
        ranked = sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0]))[:args.max_candidates]

        # Determine acceptance vs JUNK: if none over threshold, mark JUNK
        accepted = [(c, s) for c, s in ranked if s >= args.threshold]

        if not accepted:
            rows.append((raw, "JUNK", round(max(part_scores) if part_scores else 0.0, 3), "; ".join(part_explains)))
            continue

        codes = [c for c, _ in accepted]
        conf = round(float(sum(s for _, s in accepted)) / len(accepted), 3)
        explain = "; ".join(part_explains)
        rows.append((raw, " | ".join(codes), conf, explain))

    out = pd.DataFrame(rows, columns=["raw_specialty", "nucc_codes", "confidence", "explain"])
    out.to_csv(args.out, index=False)
    print(f"Saved {len(out)} rows to {args.out}")
    print(out.head(12).to_string(index=False))

if __name__ == "__main__":
    main()
