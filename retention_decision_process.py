import re
import math
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Helpers
# ----------------------------

DAN_GS_REGEX = re.compile(r'^\s*GS\b', re.IGNORECASE)
# Path to your feedback CSV
FEEDBACK_CSV_PATH = Path("dan_feedback_log.csv")

# Load feedback_df globally
if FEEDBACK_CSV_PATH.exists():
    feedback_df = pd.read_csv(FEEDBACK_CSV_PATH)
else:
    feedback_df = pd.DataFrame() 

def normalize(text: str) -> str:
    return (text or "").strip().lower()

def tokenize(s: str) -> list:
    return re.findall(r"[A-Za-z0-9]+", s.lower())

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    i = len(a & b)
    u = len(a | b)
    return i / u if u else 0.0

def normalize_dan_num(s: str) -> str:
    """Digits-only DAN number normalization."""
    return re.sub(r'\D', '', s or '')

def contains_partial_dan(user_kw: str, full_dan: str) -> bool:
    a = normalize_dan_num(user_kw)
    b = normalize_dan_num(full_dan)
    return bool(a) and (a in b)

def safe_cosine(u: np.ndarray, v: np.ndarray) -> float:
    up = np.linalg.norm(u)
    vp = np.linalg.norm(v)
    if up == 0 or vp == 0:
        return 0.0
    return float(np.dot(u, v) / (up * vp))

def _default_builder_from_df(retention_df, dan_id, sem_score, kw_set, source="keyword_rerank"):
    row = retention_df.loc[retention_df["dan"] == dan_id].head(1)
    if row.empty:
        return {"dan": dan_id, "score": sem_score, "source": source, "user_keywords": sorted(list(kw_set))}
    r = row.iloc[0]
    return {
        "dan": dan_id,
        "dan_title": r.get("dan_title",""),
        "dan_description": r.get("dan_description",""),
        "dan_category": r.get("dan_category",""),
        "dan_retention": r.get("dan_retention",""),
        "dan_designation": r.get("dan_designation",""),
        "source_pdf": r.get("source_pdf",""),
        "user_keywords": sorted(list(kw_set)),   # <- from kw_set only
        "match_score": sem_score,                      # keep original semantic score visible
        "source": source,                        # 'keyword_rerank' vs 'semantic'
    }

# ----------------------------
# 1) Keyword-only search
# ----------------------------
def keyword_search_match(
    retention_df: pd.DataFrame,
    candidate_dans: list[str],
    semantic_score: dict[str, float], 
    user_keywords: list[str],
    top_k: int = 3,
    return_builder=None,
):
    """
    Keyword search across ONLY:
      - retention_df columns: ['dan', 'dan_title', 'dan_description', 'dan_category']
      - feedback_df column:  ['user_keyword'] (if present)
    Returns the SAME schema as your regular match_retention via `return_builder`.
    """

    # Prepare fields (restrict strictly as requested)
    cols = ['dan', 'dan_title', 'dan_description', 'dan_category']
    for c in cols:
        if c not in retention_df.columns:
            retention_df[c] = ""

    kw_tokens = set(tokenize(" ".join([k for k in user_keywords if isinstance(k, str)])))

    if not kw_tokens:
        # No keywords -> empty result
        return []

    # Optional: pull historical keywords for soft expansion (kept minimal; still "nothing else")
    hist_kw = set()
    if feedback_df is not None and 'user_keyword' in feedback_df.columns:
        hist_kw = set(tokenize(" ".join(map(str, feedback_df['user_keyword'].dropna().astype(str)))))

    # Simple scoring: count overlaps of tokens across selected fields
    scores = []
    subdf = retention_df[retention_df["dan"].astype(str).isin(candidate_dans)]

    for _, row in subdf.iterrows():
        dan = str(row["dan"])
        text_blob = " ".join([str(row[c]) for c in cols])
        toks = set(tokenize(text_blob))
        score_direct = len(kw_tokens & toks)
        score_hist   = len(kw_tokens & hist_kw & toks)  # tiny nudge if keyword seen historically

        score = score_direct + 0.25 * score_hist
        if score > 0:
            scores.append((row['dan'], float(score)))

    if not scores:
        return []

    # Rank and package
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:top_k]
    print(top)

    # Build return rows using your existing format
    # Provide a builder hook so you can keep your exact schema.        
    def default_builder(dan_id, score):
        return _default_builder_from_df(retention_df, dan_id, semantic_score.get(dan_id, 0.0), kw_tokens, source="keyword_rerank")

    rb = return_builder or default_builder
    return [rb(dan, sc) for dan, sc in top]

# ----------------------------
# 2) Regular semantic matcher
# ----------------------------
def match_retention(
    retention_df: pd.DataFrame,
    doc_text: str,
    summary_text: str | None,
    embed_fn,                       # <- your offline embedding function
    schedule_embed_cache: dict,     # dict: dan -> embedding (np.ndarray), precomputed
    top_k: int = 3,
    w_semantic: float = 1.0,        # base semantic weight
    w_feedback_freq: float = 0.25,  # boost for frequently used user_dan
    w_keyword_overlap: float = 0.15,# boost from keyword overlap with this DAN row text
    w_summary_overlap: float = 0.10,# boost from summary overlap with this DAN row text
    gs_penalty: float = 0.10,       # subtract this from score if dan starts with "GS"
    dan_text_cols: tuple = ("dan_title", "dan_description", "dan_category"),
    return_builder=None,
):
    """
    Regular semantic similarity with:
      - Focus on dan, dan_title, dan_description, dan_category
      - De-prioritize DANs starting with 'GS'
      - Prioritize DANs with high frequency in feedback_df['user_dan']
      - Additional boosts using user_keywords and summary_text overlaps
    Returns: SAME schema as before via `return_builder`.
    """

    # Guard columns
    base_cols = ["dan", *dan_text_cols]
    for c in base_cols:
        if c not in retention_df.columns:
            retention_df[c] = ""

    # Build the document representation
    doc_pieces = [doc_text or ""]
   
    if summary_text:
        doc_pieces.append(summary_text)
    doc_all = " ".join(doc_pieces)
    doc_vec = embed_fn(doc_all)

    # Feedback frequency of user_dan
    freq_boost = defaultdict(int)
    if feedback_df is not None and 'user_dan' in feedback_df.columns:
        counts = Counter([str(x).strip() for x in feedback_df['user_dan'].dropna().astype(str)])
        if counts:
            max_c = max(counts.values())
            for dan_id, cnt in counts.items():
                # normalized [0,1]
                freq_boost[dan_id] = cnt / max_c if max_c else 0

    # Prepare keyword/summary token sets for overlap boosts
    if feedback_df is not None and not feedback_df.empty and 'user_keyword' in feedback_df.columns:
        kw_set = set(tokenize(" ".join(map(str, feedback_df['user_keyword'].dropna().astype(str)))))
    else:
        kw_set = set()

    sum_set = set(tokenize(summary_text or ""))

    # Optional: direct partial DAN reference boost from user keywords
    # If user enters a numeric fragment of a DAN, bonus that DAN.
    dan_partial_bonus = defaultdict(float)
    for _, row in retention_df.iterrows():
        dan_full = str(row["dan"])
        for kw in (kw_set or []):
            if isinstance(kw, str) and contains_partial_dan(kw, dan_full):
                dan_partial_bonus[dan_full] += 0.25  # modest but meaningful nudge

    # Score each DAN
    scored = []
    for _, row in retention_df.iterrows():
        dan_id = str(row["dan"])
        # Build text for this DAN (focus columns)
        dan_text = " ".join([str(row[c]) for c in dan_text_cols])
        # Use precomputed schedule embeddings (faster); fall back to on-the-fly if missing
        dan_vec = schedule_embed_cache.get(dan_id)
        if dan_vec is None:
            dan_vec = embed_fn(dan_text)
            schedule_embed_cache[dan_id] = dan_vec

        s_sem = safe_cosine(doc_vec, dan_vec)  # 0..1 (usually)
        score = w_semantic * s_sem

        # Penalize GS-prefixed DANs (e.g., "GS 09022")
        if DAN_GS_REGEX.match(dan_id):
            score -= gs_penalty

        # Feedback frequency boost
        score += w_feedback_freq * float(freq_boost.get(dan_id, 0.0))

        # Overlaps with keywords/summary (on the DAN text fields only)
        dan_tokens = set(tokenize(dan_text))
        score += w_keyword_overlap * jaccard(kw_set, dan_tokens)
        score += w_summary_overlap * jaccard(sum_set, dan_tokens)

        # Partial-DAN bonus from user keywords (digits-only containment)
        score += dan_partial_bonus.get(dan_id, 0.0)

        scored.append((dan_id, float(score)))

    # Rank
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:top_k]

    # Build return rows using your existing schema
    def default_builder(dan_id, score):
        row = retention_df.loc[retention_df['dan'] == dan_id].head(1)
        if row.empty:
            return {"dan": dan_id, "score": score}
        r = row.iloc[0]
        return {
            "dan": dan_id,
            "dan_title": r.get("dan_title", ""),
            "dan_description": r.get("dan_description", ""),
            "dan_category": r.get("dan_category", ""),
            "dan_retention": r.get("dan_retention", ""),
            "dan_designation": r.get("dan_designation", ""),
            "source_pdf": r.get("source_pdf", ""),   
            "match_score": score,
            "keyword": sorted(list(kw_set)), 
            "source": "semantic",
        }

    rb = return_builder or default_builder
    return [rb(dan, sc) for dan, sc in top]
