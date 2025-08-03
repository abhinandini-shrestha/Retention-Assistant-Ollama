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
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_embedding_model()

# Extract retention rules from PDF
def extract_retention_from_pdf(pdf_file):
    rules = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                for row in table[1:]:
                    if row and len(row) >= 3:
                        category = row[0] if row[0] else ""
                        description = row[1] if row[1] else ""
                        retention = row[2] if row[2] else ""
                        rules.append({
                            "category_name": category.strip(),
                            "category_description": description.strip(),
                            "retention_period": retention.strip()
                        })
    return pd.DataFrame(rules)

# Extract text from various file types
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return "\n".join([page.get_text() for page in doc])
    elif uploaded_file.name.endswith(".docx"):
        text = docx2txt.process(uploaded_file)
        if text.strip():  # If actual text exists
            return text

        # OCR fallback: extract images from .docx and run OCR
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
        return (
            len(line) < 5 or
            sum(c.isalpha() for c in line) / max(len(line), 1) < 0.3
        )

    filtered = [line for line in cleaned_lines if not is_low_value(line)]

    # Remove repeated lines (common headers/footers)
    line_counts = Counter(filtered)
    rare_lines = [line for line in filtered if line_counts[line] < 3]

    return "\n".join(rare_lines)
    
# Match document text to best retention category
def match_retention(text, rules_df, top_n=3):
    doc_embedding = model.encode(text, convert_to_tensor=True)
    rule_embeddings = model.encode(rules_df['category_description'].tolist(), convert_to_tensor=True)
    similarities = util.cos_sim(doc_embedding, rule_embeddings)[0]

    # Add a keyword match boost (optional hybrid scoring)
    boosted_scores = []
    for idx, row in rules_df.iterrows():
        score = float(similarities[idx])
        if any(word.lower() in text.lower() for word in row['category_description'].split()):
            score += 0.05  # small bonus for keyword overlap
        boosted_scores.append((score, idx))

    top_indices = sorted(boosted_scores, reverse=True)[:top_n]
    
    matches = []
    for score, idx in top_indices:
        matches.append({
            "category": rules_df.iloc[idx]['category_name'],
            "description": rules_df.iloc[idx]['category_description'],
            "retention": rules_df.iloc[idx]['retention_period'],
            "source_file": rules_df.iloc[idx].get('source_file', 'N/A'),
            "score": round(score, 3)
        })

    return matches

def match_retention_hybrid(text, rules_df, top_n=3):
    # Lowercased doc text for keyword matching
    text_lower = text.lower()
    
    # Step 1: Tokenize doc text for simple keyword match
    doc_tokens = set(re.findall(r'\b\w+\b', text_lower))

    # Step 2: Precompute rule embeddings
    rule_embeddings = model.encode(rules_df['category_description'].tolist(), convert_to_tensor=True)
    doc_embedding = model.encode(text, convert_to_tensor=True)
    similarities = util.cos_sim(doc_embedding, rule_embeddings)[0]

    scores = []
    for idx, row in rules_df.iterrows():
        desc = row['category_description']
        desc_lower = desc.lower()
        desc_tokens = set(re.findall(r'\b\w+\b', desc_lower))

        # Simple keyword overlap ratio
        overlap = len(doc_tokens & desc_tokens) / max(len(desc_tokens), 1)

        # Embed-based similarity
        sim_score = float(similarities[idx])

        # Final hybrid score: weight keyword overlap heavily
        final_score = (0.7 * overlap) + (0.3 * sim_score)
        scores.append((final_score, idx))

    # Get top N matches
    top_indices = sorted(scores, reverse=True)[:top_n]

    matches = []
    for score, idx in top_indices:
        matches.append({
            "category": rules_df.iloc[idx]['category_name'],
            "description": rules_df.iloc[idx]['category_description'],
            "retention": rules_df.iloc[idx]['retention_period'],
            "source_file": rules_df.iloc[idx].get('source_file', 'N/A'),
            "score": round(score, 3)
        })

    return matches

# Confidence level mapping
def get_confidence_level(score):
    if score >= 0.8:
        return "High", "🟢 High (≥ 0.80): Confident match"
    elif score >= 0.6:
        return "Medium", "🟡 Medium (0.60–0.79): Reasonable match"
    else:
        return "Low", "🔴 Low (< 0.60): Needs review"

def file_for_testing(is_scheduler):
    if is_scheduler:
        static_doc_paths = [
            "retention-schedule.pdf",
            "DOL-general.pdf"
        ]
    else:
        static_doc_paths = [
            "Testing-document-2.pdf"
        ]

    uploaded_files = []
    for path in static_doc_paths:
        with open(path, "rb") as f:
            # Mimic uploaded file structure
            uploaded_files.append(type("File", (), {
                "name": os.path.basename(path),
                "read": lambda f=f: f.read(),
                "type": f"application/{os.path.splitext(path)[1][1:]}"
            })())



# Upload and parse retention schedule
#retention_files = file_for_testing(is_scheduler=True)
   
retention_files = st.file_uploader("📎 Upload Retention Schedules (PDF)", type="pdf", accept_multiple_files=True)
if retention_files:
    all_rules = []
    for file in retention_files:
        extracted = extract_retention_from_pdf(file)
        extracted["source_file"] = file.name  # Add source tracking
        all_rules.append(extracted)
    retention_df = pd.concat(all_rules, ignore_index=True)
    print(retention_df)
    st.success(f"✅ Loaded {len(retention_df)} total retention categories from {len(retention_files)} schedule(s).")
    
    #uploaded_files = file_for_testing(is_scheduler=False)
    

    # Upload multiple documents
    uploaded_files = st.file_uploader("📄 Upload Documents to Classify (PDF, DOCX, TXT, Images)", type=["pdf", "docx", "txt", "png", "jpg"], accept_multiple_files=True)

    if uploaded_files:
        log_entries = []
        for uploaded_file in uploaded_files:

            with st.expander(f"📄 Document: {uploaded_file.name}", expanded=True):
                text = extract_text(uploaded_file)
                text = clean_document_text(text)
                
                if not text.strip():
                    st.warning("⚠️ No readable text found.")
                    continue

                match = match_retention_hybrid(text, retention_df)
                level, definition = get_confidence_level(match[0]['score'])

                st.markdown(f"**Suggested Category**: {match[0]['category']}")
                st.markdown(f"**Retention Period**: {match[0]['retention']}")
                st.markdown(f"**Description**: {match[0]['description']}")
                st.markdown(f"**Confidence Level**: {definition}")
                st.markdown(f"**Schedule Used**: {match[0]['source_file']}")

                confirm_key = f"confirm_{uploaded_file.name}"
                edit_key = f"edit_{uploaded_file.name}"

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Confirm", key=confirm_key):
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_entries.append({
                            "timestamp": timestamp,
                            "document": uploaded_file.name,
                            "predicted_category": match['category'],
                            "predicted_retention": match['retention'],
                            "override_category": match['category'],
                            "override_retention": match['retention'],
                            "confidence_level": level,
                            "confidence_score": match['score'],
                            "retention_schedule": retention_files.name
                        })
                        st.success(f"💾 Confirmed and saved for {uploaded_file.name}")

                with col2:
                    if st.button("✏️ Edit", key=edit_key):
                        override_category = st.text_input("Override Category", value=match['category'], key=f"cat_{uploaded_file.name}")
                        override_retention = st.text_input("Override Retention Period", value=match['retention'], key=f"ret_{uploaded_file.name}")
                        if st.button("💾 Save Edits", key=f"save_{uploaded_file.name}"):
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            log_entries.append({
                                "timestamp": timestamp,
                                "document": uploaded_file.name,
                                "predicted_category": match['category'],
                                "predicted_retention": match['retention'],
                                "override_category": override_category,
                                "override_retention": override_retention,
                                "confidence_level": level,
                                "confidence_score": match['score'],
                                "retention_schedule": match['source_file']
                            })
                            st.success(f"💾 Edited and saved for {uploaded_file.name}")

        if log_entries:
            log_df = pd.DataFrame(log_entries)
            log_path = "results_log.csv"
            if os.path.exists(log_path):
                existing = pd.read_csv(log_path)
                log_df = pd.concat([existing, log_df], ignore_index=True)
            log_df.to_csv(log_path, index=False)
            st.info(f"📁 Results saved to {log_path}")

            # Download button
            csv_bytes = log_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_bytes,
                file_name="results_log.csv",
                mime="text/csv"
            )
else:
    st.info("👆 Please upload a retention schedule to begin.")
