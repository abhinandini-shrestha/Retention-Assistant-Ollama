import os
import re
import tempfile
import pandas as pd
import streamlit as st
from urllib.parse import quote
import spacy
from sentence_transformers import SentenceTransformer
from generate_dan_csv import generate_csv_if_missing
from ui_helper import format_description_md

from retention_utils import (
    normalize_text,
    build_keyword_stats,
    match_retention_priority,
    extract_text,
    summarize_with_nlp
)

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

# Directory setup
retention_dir = "retention_schedules"
os.makedirs(retention_dir, exist_ok=True)

# Find available CSVs
available_csvs = sorted([
    f for f in os.listdir(retention_dir)
    if f.endswith(".csv") and os.path.isfile(os.path.join(retention_dir, f))
])

# Get all available CSVs from the retention_schedules folder
available_csvs = sorted([
    f for f in os.listdir(retention_dir)
    if f.endswith(".csv") and os.path.isfile(os.path.join(retention_dir, f))
], key=lambda f: os.path.getctime(os.path.join(retention_dir, f)), reverse=True)

st.sidebar.subheader("📁 Retention Schedule Directory")

# 1️⃣ Show clickable list
for csv in available_csvs:
    label = f"📄 {csv}"
    if st.sidebar.button(label):
        st.session_state["selected_csv"] = csv

# 2️⃣ Select default = latest, if not already selected
if "selected_csv" not in st.session_state and available_csvs:
    st.session_state["selected_csv"] = available_csvs[0]

# 3️⃣ Load selected CSV if it exists
selected_csv = st.session_state.get("selected_csv")
retention_df = None
if selected_csv:
    full_csv_path = os.path.join(retention_dir, selected_csv)
    if os.path.exists(full_csv_path):
        retention_df = pd.read_csv(full_csv_path)

# ☑️ Use previous schedule toggle
use_previous_schedule = st.checkbox("Use previously generated retention schedule", value=True)

retention_df = None
all_rules = []

# If user opts not to use previous → show upload option
if not use_previous_schedule:
    retention_files = st.file_uploader("📄 Upload Retention Schedules (PDF)", type="pdf", accept_multiple_files=True)

    if retention_files:
        for file in retention_files:
            csv_path = generate_csv_if_missing(file)
            if os.path.exists(csv_path):
                extracted = pd.read_csv(csv_path)
                if not extracted.empty:
                    extracted["source_file"] = file.name
                    all_rules.append(extracted)

        if all_rules:
            retention_df = pd.concat(all_rules, ignore_index=True)
            st.success(f"✅ Loaded {len(retention_df)} categories from uploaded schedule(s).")

# If using previous and a CSV is selected from the sidebar
elif selected_csv:
    full_csv_path = os.path.join(retention_dir, selected_csv)
    if os.path.exists(full_csv_path):
        retention_df = pd.read_csv(full_csv_path)
        st.info(f"📄 `{selected_csv}`")

# Show preview if we have a DataFrame
if retention_df is not None and not retention_df.empty:

    keyword_stats = build_keyword_stats(retention_df)

    uploaded_files = st.file_uploader("📄 Upload Documents to Classify", type=["pdf", "docx", "txt", "png", "jpg"], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.expander(f"📄 Document: {uploaded_file.name}", expanded=True):
                text = extract_text(uploaded_file)
                if not text.strip():
                    st.warning("📄 No readable text found.")
                    continue

                summary = summarize_with_nlp(text, nlp)
                matches = match_retention_priority(text, retention_df, model, keyword_stats)
                top = matches[0]

                st.markdown(f"**🥇Top Match**:")
                st.markdown(f"**DAN**: {top['dan']}")
                st.markdown(format_description_md(top["description"]))
                st.markdown(f"**Retention**: {top['retention_period']}")
                st.markdown(f"**Designation**: {top['designation']}")
                st.markdown(f"**Accuracy**: {top['match_score']}")

                with st.expander("🔍 Summary and Top Matches"):
                    st.markdown("**Document Summary:**")
                    st.code(summary)
                    st.markdown("**Top 3 Matches:**")
                    for m in matches:
                        st.markdown(f"- **{m['dan']}** ({m['retention_period']}) — Score: {m['match_score']:.2f}")
else:
    st.warning("⚠️ No retention schedule loaded. Upload a file or enable reuse above.")
