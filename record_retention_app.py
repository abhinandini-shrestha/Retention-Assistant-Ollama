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
from collections import Counter

st.set_page_config(page_title="📁 Washington Records Retention Assistant")
st.title(":file_folder: Washington Records Retention Assistant")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-distilroberta-v1")

model = load_embedding_model()

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
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    return "\n".join(lines)

def extract_retention_from_pdf(pdf_file):
    rules = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages):
            if page_num < 4:
                continue
            table = page.extract_table()
            if not table or len(table) < 4:
                continue
            for row in table[3:]:
                if not row or len(row) < 4:
                    continue
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

def build_keyword_stats(rules_df):
    all_words = []
    for desc in rules_df['description']:
        words = re.findall(r'\b\w+\b', desc.lower())
        all_words.extend(words)
    return Counter(all_words)

def match_retention_priority(text, schedule_df, model, keyword_stats, top_n=3):
    from sentence_transformers import util

    def rules_in_uploaded_doc(text):
        lines = text.splitlines()
        subject = re.search(r"Subject:\s*(.+)", text, re.IGNORECASE)
        subject_text = subject.group(1).strip() if subject else ""
        header = lines[0] if lines else ""
        first_lines = " ".join(lines[:3])
        return {
            "subject": subject_text,
            "header": header,
            "first_lines": first_lines,
            "all_text": text.lower(),
            "signals": [subject_text, header, first_lines]
        }

    def rules_in_scheduling_doc(entry):
        description = entry.get("description", "")
        full_description = description.strip().lower()
        match = re.match(r"^([^\n:\u2013\-]+)", description.strip())
        italic_title = match.group(1).strip().lower() if match else ""
        includes = re.findall(r'Includes.*?:\s*(.*?)(?:\.|\n|$)', description, re.IGNORECASE)
        excludes = re.findall(r'Excludes.*?:\s*(.*?)(?:\.|\n|$)', description, re.IGNORECASE)
        clean = lambda s: [x.strip().lower() for x in re.split(r'[,\u2022\n]', s) if x.strip()]
        includes_flat = sum([clean(i) for i in includes], [])
        excludes_flat = sum([clean(e) for e in excludes], [])
        return {
            "italic_title": italic_title,
            "includes": includes_flat,
            "excludes": excludes_flat,
            "full_description": full_description
        }

    doc_rules = rules_in_uploaded_doc(text)
    doc_words = set(re.findall(r'\b\w+\b', text.lower()))
    subject_line = doc_rules.get("subject", "").lower()
    scored_matches = []

    subject_embedding = model.encode(subject_line, convert_to_tensor=True) if subject_line else None
    doc_embedding = model.encode([text], convert_to_tensor=True)

    for _, entry in schedule_df.iterrows():
        sched_rules = rules_in_scheduling_doc(entry)
        desc_words = set(re.findall(r'\b\w+\b', sched_rules["full_description"]))
        title_words = set(re.findall(r'\b\w+\b', sched_rules["italic_title"]))

        if subject_line and any(excl in subject_line for excl in sched_rules["excludes"]):
            continue

        subject_title_sim = 0.0
        if subject_line and sched_rules["italic_title"]:
            title_embedding = model.encode([sched_rules["italic_title"]], convert_to_tensor=True)
            subject_title_sim = float(util.cos_sim(subject_embedding, title_embedding)[0][0])

        subject_title_boost = 1.0 * subject_title_sim if subject_title_sim > 0.5 else 0.0
        title_overlap = doc_words & title_words
        title_rarity = sum(2 / (keyword_stats.get(word, 0) + 1) for word in title_overlap)
        overlap = doc_words & desc_words
        rarity_score = sum(1 / (keyword_stats.get(word, 0) + 1) for word in overlap)
        entry_embedding = model.encode([sched_rules["full_description"]], convert_to_tensor=True)
        semantic_score = float(util.cos_sim(doc_embedding, entry_embedding)[0][0])

        if semantic_score > 0.97 and len(overlap) < 1:
            continue

        inclusion_bonus = 0.1 if any(
            inc in doc_rules["all_text"] for inc in sched_rules["includes"]
        ) else 0.0

        final_score = (
            1.0 * subject_title_boost +
            0.5 * title_rarity +
            0.3 * rarity_score +
            0.2 * semantic_score +
            inclusion_bonus
        )

        scored_matches.append({**entry, "match_score": round(final_score, 4)})

    return sorted(scored_matches, key=lambda x: x["match_score"], reverse=True)[:top_n]

retention_files = st.file_uploader(f"Upload Retention Schedules", type="pdf", accept_multiple_files=True)

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

        uploaded_files = st.file_uploader(f" Upload Documents to Classify", type=["pdf", "docx", "txt", "png", "jpg"], accept_multiple_files=True)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                with st.expander(f"Document: {uploaded_file.name}", expanded=True):
                    text = extract_text(uploaded_file)
                    text = clean_document_text(text)

                    if not text.strip():
                        st.warning(f"No readable text found.")
                        continue

                    matches = match_retention_priority(text, retention_df, model, keyword_stats)
                    top = matches[0]

                    st.markdown(f"**Top Match (DAN)**: {top['dan']}")
                    st.markdown(f"**Retention**: {top['retention_period']}")
                    st.markdown(f"**Description**: {top['description']}")
                    st.markdown(f"**Designation**: {top['designation']}")
                    st.markdown(f"**Score**: {top['match_score']:.4f}")

                    with st.expander(f"Top 3 Matches"):
                        for m in matches:
                            st.markdown(f"- **{m['dan']}** ({m['retention_period']}) — Score: {m['match_score']:.2f}")
    else:
        st.error(f"No valid tables found in the uploaded retention schedule PDFs.")
else:
    st.info(f"Please upload a retention schedule to begin.")

if st.button(f"Clear Cache"):
    st.cache_resource.clear()
    st.experimental_rerun()
