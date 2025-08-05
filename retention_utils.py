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
from sentence_transformers import util, SentenceTransformer
import spacy

# Lazy-loaded global model
_nlp = None

def load_spacy_model():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

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


def match_retention(text, retention_df, model, keyword_stats=None, top_k=3):
    doc_embedding = model.encode(text, convert_to_tensor=True)

    # Match against retention schedule CSV
    retention_descriptions = retention_df["description"].astype(str).tolist()
    retention_embeddings = model.encode(retention_descriptions, convert_to_tensor=True)
    retention_similarities = util.cos_sim(doc_embedding, retention_embeddings)[0]


    retention_matches = []
    for i, row in retention_df.iterrows():
        score = float(retention_similarities[i])

        # ✅ Priority boost for first uploaded file
        upload_index = row.get("upload_index", 99)
        priority_boost = 0.03 if upload_index == 0 else 0.0

        # ❌ Penalty for general retention schedule
        source_pdf = row.get("source_pdf", "").lower()
        general_penalty = -0.05 if "state-government-general-records-retention-schedule.pdf" in source_pdf else 0.0


        retention_matches.append({
            "dan": row["dan"],
            "description": row["description"],
            "retention_period": row.get("retention_period", ""),
            "designation": row.get("designation", ""),
            "match_score": score,
            "source": "retention_schedule"
        })

    feedback_df = load_feedback_df()
    feedback_matches = []
    if feedback_df is not None and not feedback_df.empty:
        feedback_texts = feedback_df["summary_text"].astype(str).tolist()
        feedback_embeddings = model.encode(feedback_texts, convert_to_tensor=True)
        similarities = util.cos_sim(doc_embedding, feedback_embeddings)[0]
        dan_lookup = retention_df.drop_duplicates(subset="dan").set_index("dan").to_dict("index")

        for i, row in feedback_df.iterrows():
            base_score = float(similarities[i])
            if base_score > 0.95:
                boost = 1.5
            elif base_score > 0.85:
                boost = 1.25
            elif base_score > 0.8:
                boost = 1.1
            else:
                boost = 1.0

            adjusted_score = (base_score * boost) + 0.05
            meta = dan_lookup.get(row["user_dan"], {})

            feedback_matches.append({
                "dan": row["user_dan"],
                "description": meta.get("description", row["description"]),
                "retention_period": meta.get("retention_period", row.get("retention_period", "")),
                "designation": meta.get("designation", row.get("designation", "")),
                "match_score": adjusted_score,
                "source": "feedback_log"
            })

    all_matches = retention_matches + feedback_matches
    top_matches = sorted(all_matches, key=lambda x: x["match_score"], reverse=True)[:top_k]

    return top_matches

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
    nlp = load_spacy_model()
    doc = nlp(text)

    # Grab first meaningful sentence (used as title keyword)
    title_keyword = next((s.text.strip().lower() for s in doc.sents if len(s.text.strip()) > 5), "")

    # Extract key noun phrases
    noun_phrases = [chunk.text.strip().lower() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 3]

    # Include user-entered keywords
    user_kw = [w.strip().lower() for w in user_keywords.split(",")] if user_keywords else []

    # Combine all
    all_keywords = set(noun_phrases + user_kw + [title_keyword])
    return ", ".join(sorted(all_keywords))