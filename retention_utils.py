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
import yake

def load_spacy_model():
    return spacy.load("en_core_web_sm")

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

def match_retention_priority(text, schedule_df, model, keyword_stats, top_n=3, keyword_boost_weight=0.1):
    clean_text = normalize_text(text)
    doc_words = set(clean_text.split())
    doc_embedding = model.encode([clean_text], convert_to_tensor=True)

    feedback_keywords = load_feedback_keywords()

    scored_matches = []
    for _, row in schedule_df.iterrows():
        dan = row["dan"]
        desc = normalize_text(row["description"])
        desc_words = set(desc.split())

        #overlap = doc_words & desc_words
        #rarity_score = sum(1 / (keyword_stats[w] + 1) for w in overlap)

        entry_embedding = model.encode([desc], convert_to_tensor=True)
        base_score = float(util.cos_sim(doc_embedding, entry_embedding)[0][0]) #usually between 0.3–0.9

        # Boost score if user keywords overlap
        user_kw = feedback_keywords.get(dan, [])
        overlap = [kw for kw in user_kw if kw in doc_text]
        boost = keyword_boost_weight * len(overlap)

        #final_score = 0.6 * semantic_score + 0.4 * rarity_score
        scored_matches.append({**row, "match_score": round(base_score + boost, 4)})
    return sorted(scored_matches, key=lambda x: x["match_score"], reverse=True)[:top_n]

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

def summarize_with_nlp(text, nlp, max_sents=3):
    doc = nlp(text)
    return "\n".join(str(s) for s in list(doc.sents)[:max_sents])

def load_feedback_keywords(feedback_csv="dan_feedback_log.csv"):
    dan_keywords = {}
    if not os.path.exists(feedback_csv):
        return dan_keywords

    df = pd.read_csv(feedback_csv)
    grouped = df[df["user_keywords"].notna()].groupby("user_dan")["user_keywords"]
    for dan, keywords in grouped:
        all_words = []
        for kw_str in keywords.dropna():
            all_words.extend([w.strip().lower() for w in kw_str.split(",")])
        dan_keywords[dan] = list(set(all_words))
    return dan_keywords


def extract_keywords(text, nlp, user_keywords=None):
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