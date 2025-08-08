import os
import re
import io
import zipfile
import tempfile
import pytesseract
import docx2txt
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
from collections import Counter
from sentence_transformers import util
from typing import Callable, Dict, Iterable, List, Optional, Tuple

#import spacy

# Lazy-loaded global model
#_nlp = None

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def build_keyword_stats(rules_df):
    all_words = []
    for desc in rules_df['description']:
        words = re.findall(r'\b\w+\b', desc.lower())
        all_words.extend(words)
    return Counter(all_words)

def load_valid_feedback(feedback_path, retention_df):
    try:
        feedback_df = pd.read_csv(feedback_path)
    except Exception:
        return pd.DataFrame()  # return empty if not loadable

    if feedback_df.empty:
        return pd.DataFrame()

    # Drop rows with missing required fields
    required_cols = {"was_correct", "user_dan", "predicted_dan", "description", "summary_text"}
    if not required_cols.issubset(feedback_df.columns):
        return pd.DataFrame()

    # Validate correctness logic
    invalid_rows = feedback_df[
        ((feedback_df["was_correct"] == True) & (feedback_df["user_dan"] != feedback_df["predicted_dan"])) |
        ((feedback_df["was_correct"] == False) & (feedback_df["user_dan"] == feedback_df["predicted_dan"]))
    ]

    feedback_df = feedback_df.drop(index=invalid_rows.index)

    # Keep only feedback whose DAN is in current schedule
    valid_dans = set(retention_df["dan"])
    feedback_df = feedback_df[feedback_df["user_dan"].isin(valid_dans)]

    return feedback_df


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

DAN_REGEX = re.compile(r"\b\d{2}-\d{2}-\d{5}\b")  # e.g., 06-02-61108

def match_retention(
    text: str,
    retention_df: pd.DataFrame,
    model,
    keyword_stats: Optional[Dict[str, float]] = None,

    top_k: int = 3,
    feedback_loader: Optional[Callable[[], Optional[pd.DataFrame]]] = None,
) -> List[Dict]:
    if retention_df is None or retention_df.empty:
        return []
    print(keyword_stats)

    # Column mapping (case-insensitive)
    cols = {c.lower(): c for c in retention_df.columns}
    dan_col  = cols.get("dan")
    desc_col = cols.get("description") or cols.get("category_description")
    rp_col   = cols.get("retention_period", "retention_period")
    des_col  = cols.get("designation", "designation")

    # ---- 1) Exact DAN short-circuit if a DAN-like keyword is present ----
    user_keywords = list((keyword_stats or {}).keys()) if isinstance(keyword_stats, dict) else []
    print(user_keywords)
    dan_like = [k.strip() for k in user_keywords if isinstance(k, str) and DAN_REGEX.fullmatch(k.strip())]

    if dan_like and dan_col:
        dan_norm = set(retention_df[dan_col].astype(str).str.strip().str.lower())
        exact_hits = [d for d in dan_like if d.lower() in dan_norm]
        if exact_hits:
            rows = retention_df[retention_df[dan_col].astype(str).str.strip().str.lower().isin([d.lower() for d in exact_hits])].head(top_k)
            return [{
                "dan": r.get(dan_col, ""),
                "description": r.get(desc_col, r.get("category_description", "")),
                "retention_period": r.get(rp_col, ""),
                "designation": r.get(des_col, ""),
                "match_score": 999.0,
                "source": "dan_exact",
            } for _, r in rows.iterrows()]

    # ---- 2) Semantic path (retention + feedback) ----
    doc_emb = model.encode(text, convert_to_tensor=True)

    # Retention schedule similarities
    basis_texts = retention_df[desc_col].astype(str).tolist() if desc_col else retention_df.iloc[:, 0].astype(str).tolist()
    basis_embs  = model.encode(basis_texts, convert_to_tensor=True)
    sims        = util.cos_sim(doc_emb, basis_embs)[0]

    ret_matches = []
    for i, row in retention_df.iterrows():
        score = float(sims[i])
        # + priority boost for first uploaded file
        if row.get("upload_index", 99) == 0:
            score += 0.03
        # - penalty for general schedule
        if "state-government-general-records-retention-schedule.pdf" in str(row.get("source_pdf", "")).lower():
            score -= 0.05

        ret_matches.append({
            "dan": row.get(dan_col, ""),
            "description": row.get(desc_col, row.get("category_description", "")),
            "retention_period": row.get(rp_col, ""),
            "designation": row.get(des_col, ""),
            "match_score": score,
            "source": "retention_schedule",
        })

    # Feedback boosts (optional)
    fb_matches = []
    fb_df = feedback_loader() if feedback_loader else _try_load_feedback_df()
    if fb_df is not None and not fb_df.empty:
        fb_texts = fb_df["summary_text"].astype(str).tolist()
        fb_embs  = model.encode(fb_texts, convert_to_tensor=True)
        fb_sims  = util.cos_sim(doc_emb, fb_embs)[0]

        dan_lookup = {}
        if dan_col:
            dan_lookup = retention_df.drop_duplicates(subset=dan_col).set_index(dan_col).to_dict("index")

        for i, r in fb_df.iterrows():
            base = float(fb_sims[i])
            boost = 1.5 if base > 0.95 else 1.25 if base > 0.85 else 1.1 if base > 0.8 else 1.0
            score = (base * boost) + 0.05

            udan = str(r.get("user_dan", "") or "")
            meta = dan_lookup.get(udan, {})
            fb_matches.append({
                "dan": udan,
                "description": meta.get(desc_col, r.get("description", "")),
                "retention_period": meta.get(rp_col, r.get("retention_period", "")),
                "designation": meta.get(des_col, r.get("designation", "")),
                "match_score": score,
                "source": "feedback_log",
            })

    all_matches = ret_matches + fb_matches
    return sorted(all_matches, key=lambda x: x["match_score"], reverse=True)[:top_k]


def _try_load_feedback_df() -> Optional[pd.DataFrame]:
    """Attempt to import a host-app loader; return None if unavailable."""
    try:
        from __main__ import load_feedback_df  # provided by host app
        return load_feedback_df()
    except Exception:
        return None



def extract_text_from_pdf_images(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"])).convert("L")
            image = image.filter(ImageFilter.MedianFilter())
            image = ImageEnhance.Contrast(image).enhance(2.0)
            text = pytesseract.image_to_string(image, config="--psm 6")
            if text.strip():
                all_text.append(text.strip())
    return "\n\n".join(all_text)

def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        text = ""
        file_bytes = uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(file_bytes)
            tmpfile_path = tmpfile.name

        with fitz.open(tmpfile_path) as doc:
            for page in doc:
                page_text = page.get_text("text").strip()
                if page_text:
                    text += page_text + "\n"

        ocr_text = extract_text_from_pdf_images(tmpfile_path)
        if ocr_text:
            text += "\n" + ocr_text

        os.remove(tmpfile_path)
        return text

    elif uploaded_file.name.endswith(".docx"):
        text = docx2txt.process(uploaded_file)
        if text.strip():
            return text
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                image_texts = []
                for file in zip_ref.namelist():
                    if file.startswith("word/media/") and file.endswith((".png", ".jpg", ".jpeg")):
                        zip_ref.extract(file, tmpdir)
                        img = Image.open(os.path.join(tmpdir, file))
                        image_texts.append(pytesseract.image_to_string(img))
                return "\n".join(image_texts)

    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    elif uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)

    return ""

def summarize_with_nlp(text, max_sents=3):
    nlp = load_spacy_model()
    doc = nlp(text)

    return "\n".join(str(s) for s in list(doc.sents)[:max_sents])

def load_feedback_df(path="dan_feedback_log.csv"):
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def extract_keywords(text, user_keywords=None):
    # Clean and tokenize basic words
    text = text.lower()
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text)  # only words 4+ letters

    # Get top unique keywords (limited to first N words to keep it simple)
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    sorted_keywords = sorted(word_freq, key=word_freq.get, reverse=True)
    top_keywords = sorted_keywords[:10]  # top 10 frequent words

    # Include user keywords
    user_kw = [w.strip().lower() for w in user_keywords.split(",")] if user_keywords else []

    # Combine and deduplicate
    all_keywords = sorted(set(top_keywords + user_kw))
    return ", ".join(all_keywords)

def summarize_without_nlp(text, max_sents=3):
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    if not sentences:
        return "No summary available."

    # Look for sentences containing certain useful keywords
    priority_keywords = ["summary", "purpose", "includes", "covers", "details"]
    prioritized = [s for s in sentences if any(k in s.lower() for k in priority_keywords)]

    selected = prioritized[:max_sents] if prioritized else sentences[:max_sents]
    return " ".join(selected)

def load_dan_catalog(csv_path: str) -> pd.DataFrame | None:
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        # expected helpful columns: DAN, Title, Retention, Designation, Category Description
        return df
    except Exception:
        return None