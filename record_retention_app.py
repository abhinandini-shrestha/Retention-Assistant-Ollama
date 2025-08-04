import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import tempfile
import pandas as pd
import streamlit as st
from urllib.parse import quote
from generate_dan_csv import generate_csv_if_missing, extract_keywords
from ui_helper import format_description_md, render_read_only_view, render_editable_form
import csv
from datetime import datetime

from retention_utils import (
    normalize_text,
    build_keyword_stats,
    match_retention_priority,
    extract_text,
    summarize_with_nlp,
    load_spacy_model,
    load_embedding_model
)

st.set_page_config(page_title="📁 Washington Records Retention Assistant")
st.title("📁 Washington Records Retention Assistant")

# Load models
@st.cache_resource
def get_nlp():
    return load_spacy_model()

@st.cache_resource
def get_embedder():
    return load_embedding_model()

nlp = get_nlp()
model = get_embedder()

# Directory setup
retention_dir = "retention_schedules"
os.makedirs(retention_dir, exist_ok=True)

# Get all available CSVs from the retention_schedules folder
available_csvs = sorted([
    f for f in os.listdir(retention_dir)
    if f.endswith(".csv") and os.path.isfile(os.path.join(retention_dir, f))
], key=lambda f: os.path.getctime(os.path.join(retention_dir, f)), reverse=True)

st.sidebar.subheader("📁 Retention Schedule Directory")

# 1️⃣ Show clickable list
for csv_file in available_csvs:
    label = f"📄 {csv_file}"
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

    # Build DAN lookup dictionary once
    dan_lookup = retention_df.drop_duplicates(subset="dan").set_index("dan").to_dict("index")
    dan_options = list(dan_lookup.keys())

    uploaded_files = st.file_uploader("📄 Upload Documents to Classify", type=["pdf", "docx", "txt", "png", "jpg"], accept_multiple_files=True)
    if uploaded_files:
        for i, uploaded_file in enumerate(uploaded_files):

            with st.expander(f"📄 Document: {uploaded_file.name}", expanded=True):
                text = extract_text(uploaded_file)
                if not text.strip():
                    st.warning("📄 No readable text found.")
                    continue

                summary = summarize_with_nlp(text, nlp)
                matches = match_retention_priority(text, retention_df, model, keyword_stats)
                top = matches[0]

                edit_mode_key = f"edit_mode_{i}"
                edit_button_key = f"edit_button_{i}"
                confirm_button_key = f"confirm_button_{i}"

                # Initialize only once
                if edit_mode_key not in st.session_state:
                    st.session_state[edit_mode_key] = False

                # Set edit mode on edit button
                if st.button("✏️ Edit Classification", key=edit_button_key):
                    st.session_state[edit_mode_key] = True

                # Render based on session state
                if st.session_state[edit_mode_key]:
                    corrected_dan, guidance_keywords = render_editable_form(i, top, retention_df, dan_lookup)

                    if st.button("✅ Confirm Update", key=confirm_button_key):
                        # Save logic...
                        st.success("✅ Feedback saved.")

                        # Switch to read-only view after confirming
                        st.session_state[edit_mode_key] = False

                        # Show read-only view explicitly after confirm
                        render_read_only_view(top, matches)

                else:
                    render_read_only_view(top, matches)

                with st.expander("🔍 Summary and Top Matches"):
                    st.markdown("**Document Summary:**")
                    st.code(summary)

else:
    st.warning("⚠️ No retention schedule loaded. Upload a file or enable reuse above.")
