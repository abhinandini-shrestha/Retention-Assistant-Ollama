import streamlit as st
import pandas as pd
import pdfplumber
import pytesseract
import docx2txt
import fitz  # PyMuPDF
import os
import tempfile
from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageEnhance, ImageFilter
import io
import zipfile
import re
from collections import Counter
import spacy

st.set_page_config(page_title="📁 Washington Records Retention Assistant")
st.title("📁 Washington Records Retention Assistant")

# Load models
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

model = load_embedding_model()
nlp = load_spacy_model()

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

def extract_retention_from_pdf(pdf_file):
    rules = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if not table or len(table) < 2:
                continue
            for row in table[1:]:
                if len(row) < 4 or not any(row):
                    continue
                rules.append({
                    "dan": (row[0] or "").strip(),
                    "description": (row[1] or "").strip(),
                    "retention_period": (row[2] or "").strip(),
                    "designation": (row[3] or "").strip(),
                })
    return pd.DataFrame(rules)

def extract_italic_title(description):
    match = re.match(r"^([^\n:]+)", description.strip())
    return match.group(1).strip() if match else description.strip()

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

def summarize_with_nlp(text):
    doc = nlp(text)
    return "\n".join(str(s) for s in list(doc.sents)[:3])

def match_retention_priority(text, schedule_df, model, keyword_stats, top_n=3):
    clean_text = normalize_text(text)
    doc_words = set(clean_text.split())
    doc_embedding = model.encode([clean_text], convert_to_tensor=True)
    scored_matches = []
    for _, row in schedule_df.iterrows():
        desc = normalize_text(row["description"])
        desc_words = set(desc.split())
        overlap = doc_words & desc_words
        rarity_score = sum(1 / (keyword_stats[w] + 1) for w in overlap)
        entry_embedding = model.encode([desc], convert_to_tensor=True)
        semantic_score = float(util.cos_sim(doc_embedding, entry_embedding)[0][0])
        final_score = 0.6 * semantic_score + 0.4 * rarity_score
        scored_matches.append({**row, "match_score": round(final_score, 4)})
    return sorted(scored_matches, key=lambda x: x["match_score"], reverse=True)[:top_n]

retention_files = st.file_uploader("📄 Upload Retention Schedules (PDF)", type="pdf", accept_multiple_files=True)

if retention_files:
    all_rules = []
    for file in retention_files:
        extracted = extract_retention_from_pdf(file)
        if not extracted.empty:
            extracted["source_file"] = file.name
            all_rules.append(extracted)

    if all_rules:
        retention_df = pd.concat(all_rules, ignore_index=True)
        keyword_stats = build_keyword_stats(retention_df)
        st.success(f"✅ Loaded {len(retention_df)} categories.")

        uploaded_files = st.file_uploader("📄 Upload Documents to Classify", type=["pdf", "docx", "txt", "png", "jpg"], accept_multiple_files=True)
        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.expander(f"📄 Document: {uploaded_file.name}", expanded=True):
                    text = extract_text(uploaded_file)
                    if not text.strip():
                        st.warning("📄 No readable text found.")
                        continue

                    summary = summarize_with_nlp(text)
                    matches = match_retention_priority(text, retention_df, model, keyword_stats)
                    top = matches[0]

                    st.markdown(f"**Top Match (DAN)**: {top['dan']}")
                    st.markdown(f"**Retention**: {top['retention_period']}")
                    st.markdown(f"**Description**: {top['description']}")
                    st.markdown(f"**Designation**: {top['designation']}")
                    st.markdown(f"**Score**: {top['match_score']}")

                    with st.expander("🔍 Summary and Top Matches"):
                        st.markdown("**Document Summary:**")
                        st.code(summary)
                        st.markdown("**Top 3 Matches:**")
                        for m in matches:
                            st.markdown(f"- **{m['dan']}** ({m['retention_period']}) — Score: {m['match_score']:.2f}")
    else:
        st.error("❌ No valid retention data extracted.")
else:
    st.info("👆 Please upload a retention schedule to begin.")

if st.button("🔄 Clear Cache"):
    st.cache_resource.clear()
    st.experimental_rerun()
