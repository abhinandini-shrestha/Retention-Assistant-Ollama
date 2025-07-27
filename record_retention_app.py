import streamlit as st
import pandas as pd
import pdfplumber
import pytesseract
import docx2txt
import fitz  # pymupdf
import os
import tempfile
import datetime
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import io
import zipfile
import re
from xml.etree import ElementTree as ET
from collections import Counter

st.set_page_config(page_title="📁 Washington Records Retention Assistant")
st.title("📁 Washington Records Retention Assistant")

# Load embedding model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

model = load_embedding_model()

def build_keyword_stats(rules_df):
    all_words = []
    for desc in rules_df['description']:
        words = re.findall(r'\b\w+\b', desc.lower())
        all_words.extend(words)
    return Counter(all_words)

def extract_retention_from_pdf(pdf_file):
    rules = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            if page_num < 4:  # Skip first 4 pages (usually index/table of contents)
                continue

            table = page.extract_table()
            if not table or len(table) < 4:
                continue  # Not enough rows to contain data

            for row in table[3:]:  # Skip first 3 rows (title rows)
                if not row or len(row) < 4:
                    continue  # Skip incomplete rows

                dan = row[0] or ""
                description = row[1] or ""
                retention = row[2] or ""
                designation = row[3] or ""

                rules.append({
                    "dan": dan.strip(),
                    "description": description.strip(),
                    "retention_period": retention.strip(),
                    "designation": designation.strip(),
                })

    return pd.DataFrame(rules)

def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
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
                        img_path = os.path.join(tmpdir, file)
                        img = Image.open(img_path)
                        image_texts.append(pytesseract.image_to_string(img))
                return "\n".join(image_texts)
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)
    else:
        return ""

def clean_document_text(raw_text):
    # Split into lines and remove whitespace
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

    # Remove lines from Table of Contents or Index sections
    cleaned_lines = []
    skip = False
    for line in lines:
        if re.search(r'table of contents | indexes', line, re.IGNORECASE):
            skip = True
        elif skip and re.match(r'^\s*\d+(\.\d+)*\s*$', line):  # Likely index entry
            continue
        elif skip and len(line) < 30:
            continue
        else:
            skip = False
            cleaned_lines.append(line)

    # Filter out lines that are too short or mostly digits/symbols
    def is_low_value(line):
        return len(line) < 5 or sum(c.isalpha() for c in line) / max(len(line), 1) < 0.3

    filtered = [line for line in cleaned_lines if not is_low_value(line)]

    # Remove repeated lines (common headers/footers)
    line_counts = Counter(filtered)
    rare_lines = [line for line in filtered if line_counts[line] < 3]

    return "\n".join(rare_lines)

def extract_italic_title(description):
    match = re.match(r"^([^\n:]+)", description.strip())
    return match.group(1).strip() if match else description.strip()

def match_retention_priority(text, rules_df, keyword_stats, top_n=3):
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    scores = []

    for idx, row in rules_df.iterrows():
        if 'description' not in row or not isinstance(row['description'], str):
            continue

        description = row['description']
        italic_title = extract_italic_title(description).lower()
        title_words = set(re.findall(r'\b\w+\b', italic_title))
        desc_words = set(re.findall(r'\b\w+\b', description.lower()))

        overlap = text_words & desc_words
        title_overlap = text_words & title_words

        # Score by rarity (lower frequency = higher score)
        rarity_score = sum(1 / (keyword_stats[word] + 1) for word in overlap)
        title_bonus = sum(2 / (keyword_stats[word] + 1) for word in title_overlap)

        final_score = rarity_score + title_bonus
        scores.append((final_score, idx))

    top_matches = sorted(scores, reverse=True)[:top_n]

    return [{
        "category": rules_df.iloc[idx]['dan'],
        "description": rules_df.iloc[idx]['description'],
        "retention": rules_df.iloc[idx]['retention_period'],
        "designation": rules_df.iloc[idx]['designation'],
        "score": round(score, 3)
    } for score, idx in top_matches]

retention_files = st.file_uploader(f"📄 Upload Retention Schedules (PDF)", type="pdf", accept_multiple_files=True)

if retention_files:
    all_rules = []
    for file in retention_files:
        extracted = extract_retention_from_pdf(file)
        if not extracted.empty:
            extracted["source_file"] = file.name  # Add source tracking
            all_rules.append(extracted)

    if all_rules:
        retention_df = pd.concat(all_rules, ignore_index=True)
        if "description" in retention_df.columns:
            keyword_stats = build_keyword_stats(retention_df)

            st.success(f"✅ Loaded {len(retention_df)} categories from {len(retention_files)} file(s).")

            # Continue with file uploads and classification
        else:
            st.error("❌ No 'description' column found in extracted retention data.")
    else:
        st.error("❌ No valid tables found in the uploaded retention schedule PDFs.")

    uploaded_files = st.file_uploader(f"📄  Upload Documents to Classify (PDF, DOCX, TXT, Images)", type=["pdf", "docx", "txt", "png", "jpg"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.expander(f"📄 Document: {uploaded_file.name}", expanded=True):
                text = extract_text(uploaded_file)
                text = clean_document_text(text)

                if not text.strip():
                    st.warning(f"📄  No readable text found.")
                    continue

                matches = match_retention_priority(text, retention_df, keyword_stats)
                top = matches[0]

                st.markdown(f"**Top Match (DAN)**: {top['category']}")
                st.markdown(f"**Retention**: {top['retention']}")
                st.markdown(f"**Designation**: {top['designation']}")
                st.markdown(f"**Description**: {top['description']}")
                st.markdown(f"**Score**: {top['score']}")

                with st.expander(f"📄 Top 3 Matches"):
                    for m in matches:
                        st.markdown(f"- **{m['category']}** ({m['retention']}) — Score: {m['score']}")
else:
    st.info("👆 Please upload a retention schedule to begin.")

if st.button("🔄 Clear Cache"):
    st.cache_resource.clear()
    st.experimental_rerun()
