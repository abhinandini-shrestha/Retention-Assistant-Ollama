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
from PIL import Image, ImageEnhance, ImageFilter
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
    return SentenceTransformer("all-MiniLM-L6-v2")

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

def extract_text_from_pdf_images(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("L")
            image = image.filter(ImageFilter.MedianFilter())
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            text = pytesseract.image_to_string(image, config="--psm 6")
            if text.strip():
                all_text.append(text.strip())
    return "\n\n".join(all_text)


def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.read())
            tmpfile_path = tmpfile.name
        with fitz.open(tmpfile_path) as doc:
            for page in doc:
                page_text = page.get_text().strip()
                if page_text:
                    text += page_text + "\n"
        image_text = extract_text_from_pdf_images(tmpfile_path)
        os.remove(tmpfile_path)
        print(image_text)
        return text + "\n" + image_text
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

def rules_in_scheduling_doc(entry):
    description = entry.get("description", "")
    description_lower = description.lower()

    # --- Italic title extraction ---
    # Try to extract if wrapped in underscores (e.g., "_Driver Hearings_")
    italic_match = re.search(r"_(.*?)_", description)
    if italic_match:
        italic_title = italic_match.group(1).strip()
    else:
        # Fallback: use the first phrase before a colon or newline
        match = re.match(r"^([^\n:]+)", description.strip())
        italic_title = match.group(1).strip() if match else ""

    # --- Includes ---
    includes_match = re.search(r"includes.*?:\s*(.*)", description, re.IGNORECASE)
    includes = []
    if includes_match:
        includes_raw = includes_match.group(1)
        includes = [item.strip().lower() for item in includes_raw.split(",") if item.strip()]

    # --- Excludes ---
    excludes_match = re.search(r"excludes.*?:\s*(.*)", description, re.IGNORECASE)
    excludes = []
    if excludes_match:
        excludes_raw = excludes_match.group(1)
        excludes = [item.strip().lower() for item in excludes_raw.split(",") if item.strip()]

    return {
        "italic_title": italic_title.lower(),
        "includes": includes,
        "excludes": excludes,
        "full_description": description_lower,
    }

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
        "all_text": text.lower(),  # ✅ Add this line
        "signals": [subject_text, header, first_lines]
    }


def match_retention_priority(text, schedule_df, model, keyword_stats, top_n=3):
    import numpy as np
    doc_rules = rules_in_uploaded_doc(text)
    doc_words = set(re.findall(r'\b\w+\b', text.lower()))

    subject_line = doc_rules.get("subject", "")
    scored_matches = []

    # Pre-encode document subject (semantic match)
    subject_embedding = model.encode(subject_line, convert_to_tensor=True) if subject_line else None
    doc_embedding = model.encode([text], convert_to_tensor=True)

    for _, entry in schedule_df.iterrows():
        sched_rules = rules_in_scheduling_doc(entry)
        desc_words = set(re.findall(r'\b\w+\b', sched_rules["full_description"]))
        title_words = set(re.findall(r'\b\w+\b', sched_rules["italic_title"]))

        # 1. Subject line vs italic title (highest priority semantic match)
        subject_title_sim = 0.0
        if subject_line and sched_rules["italic_title"]:
            title_embedding = model.encode([sched_rules["italic_title"]], convert_to_tensor=True)
            subject_title_sim = float(util.cos_sim(subject_embedding, title_embedding)[0][0])

        subject_title_boost = 1.0 * subject_title_sim if subject_title_sim > 0.5 else 0

        # 2. Italic title rarity score
        title_overlap = doc_words & title_words
        title_rarity = sum(2 / (keyword_stats[word] + 1) for word in title_overlap)

        # 3. General rarity score
        overlap = doc_words & desc_words
        rarity_score = sum(1 / (keyword_stats[word] + 1) for word in overlap)

        # 4. Semantic similarity (doc ↔ description)
        entry_texts = [sched_rules["full_description"]]
        entry_embedding = model.encode(entry_texts, convert_to_tensor=True)
        semantic_score = float(util.cos_sim(doc_embedding, entry_embedding)[0][0])

        # 5. Rejection rule: skip overly vague semantic matches
        if semantic_score > 0.95 and len(overlap) < 2:
            continue  # Reject: very general similarity

        # 6. Inclusion boost
        inclusion_bonus = 0.1 if any(inc in doc_rules["all_text"] for inc in sched_rules["includes"]) else 0.0

        # 7. Exclusion penalty
       # exclusion_penalty = 0.5 if any(exc in doc_rules["subject"] for exc in sched_rules["excludes"]) else 1.0

        # --- Final Score Composition ---
        final_score = (
            1.0 * subject_title_boost +   # Highest weight
            1.0 * title_rarity +
            0.3 * rarity_score +
            0.2 * semantic_score +
            inclusion_bonus
        )

        scored_matches.append({**entry, "match_score": round(final_score, 4)})

    return sorted(scored_matches, key=lambda x: x["match_score"], reverse=True)[:top_n]

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

                matches = match_retention_priority(text, retention_df, model, keyword_stats)
                top = matches[0]

                st.markdown(f"**Top Match (DAN)**: {top['dan']}")
                st.markdown(f"**Retention**: {top['retention_period']}")
                st.markdown(f"**Description**: {top['description']}")
                st.markdown(f"**Designation**: {top['designation']}")
                st.markdown(f"**Score**: {top['match_score']}")

                with st.expander(f"📄 Top 3 Matches"):
                    for m in matches:
                        st.markdown(f"- **{m['dan']}** ({m['retention_period']}) — Score: {m['match_score']:.2f}")
else:
    st.info("👆 Please upload a retention schedule to begin.")

if st.button("🔄 Clear Cache"):
    st.cache_resource.clear()
    st.experimental_rerun()
