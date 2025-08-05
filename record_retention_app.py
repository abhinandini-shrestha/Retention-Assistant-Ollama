import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import tempfile
import pandas as pd
import streamlit as st
from urllib.parse import quote
from generate_dan_csv import generate_csv_if_missing, extract_keywords, list_uploaded_sources, clear_retention_schedules
from ui_helper import format_description_md, render_read_only_view, render_editable_form
import csv
from datetime import datetime

from retention_utils import (
    normalize_text,
    build_keyword_stats,
    match_retention,
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

# Session state tracking
if "cleared_on_checked" not in st.session_state:
    st.session_state["cleared_on_checked"] = False

if "last_uploaded_files" not in st.session_state:
    st.session_state["last_uploaded_files"] = set()

# Directory setup
retention_dir = "retention_schedules"
os.makedirs(retention_dir, exist_ok=True)

# Get all available CSVs from the retention_schedules folder
available_csvs = sorted([
    f for f in os.listdir(retention_dir)
    if f.endswith(".csv") and os.path.isfile(os.path.join(retention_dir, f))
], key=lambda f: os.path.getctime(os.path.join(retention_dir, f)), reverse=True)

# st.sidebar.subheader("📁 Retention Schedule Directory")
#
# # 1️⃣ Show clickable list
# for filename, source_pdf, date in list_uploaded_sources():
#     st.sidebar.markdown(f"- **{source_pdf}**<br><small>{date[:19]}</small>", unsafe_allow_html=True)
#
# # 2️⃣ Select default = latest, if not already selected
# if "selected_csv" not in st.session_state and available_csvs:
#     st.session_state["selected_csv"] = available_csvs[0]
#
# # 3️⃣ Load selected CSV if it exists
# selected_csv = st.session_state.get("selected_csv")
# retention_df = None
# if selected_csv:
#     full_csv_path = os.path.join(retention_dir, selected_csv)
#     if os.path.exists(full_csv_path):
#         retention_df = pd.read_csv(full_csv_path)

# ☑️ Use previous schedule toggle
use_previous_schedule = st.checkbox("Use previous retention schedule", value=True)

retention_df = None
all_rules = []
new_upload = False
retention_files = []

# If user opts not to use previous → show upload option
if not use_previous_schedule:
    st.session_state["cleared_on_checked"] = False
    retention_files = st.file_uploader("📄 Upload Retention Schedules (PDF)", type="pdf", accept_multiple_files=True)

    uploaded_names = set(f.name for f in retention_files) if retention_files else set()
    if uploaded_names != st.session_state["last_uploaded_files"]:
        new_upload = True
        st.session_state["last_uploaded_files"] = uploaded_names

    if new_upload:
        clear_retention_schedules()
        st.session_state["cleared_on_checked"] = True

    if retention_files:
        for idx, file in enumerate(retention_files):
            csv_path = generate_csv_if_missing(file)
            if csv_path is not None:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    df["source_pdf"] = file.name
                    df["upload_date"] = datetime.now().isoformat()
                    df["upload_index"] = idx
                    df.to_csv(csv_path, index=False)  # update CSV with metadata
                    all_rules.append(df)

        if all_rules:
            retention_df = pd.concat(all_rules, ignore_index=True)
            st.success(f"✅ Loaded {len(retention_df)} categories from {file.name}.")

# If using previous and a CSV is selected from the sidebar
else:
    all_dfs = []
    csv_files = [f for f in os.listdir(retention_dir) if f.endswith(".csv")]
    for f in csv_files:
        csv_path = os.path.join(retention_dir, f)
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                all_dfs.append(df)
                source = df['source_pdf'].iloc[0] if 'source_pdf' in df.columns else "Unknown"
                date = df['upload_date'].iloc[0][:19] if 'upload_date' in df.columns else "Unknown"

                # Short text preview of first 3 rows

                st.markdown(
                    f"""
                <div style="background-color:#f0f8ff; padding:1rem; border-left:5px solid #1e90ff; border-radius:0.5rem; margin-bottom:1rem;">
                <b>Source PDF:</b> {source}<br>
                <b>Uploaded on:</b> {date}
                </div>
                """,
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.warning(f"⚠️ Could not read `{f}`: {e}")

        if all_dfs:
                retention_df = pd.concat(all_dfs, ignore_index=True)


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

                summary = summarize_with_nlp(text)
                matches = match_retention(text, retention_df, model, keyword_stats)
                top = matches[0]

                edit_mode_key = f"edit_mode_{i}"
                edit_button_key = f"edit_button_{i}"
                confirm_button_key = f"confirm_button_{i}"

                # Initialize only once
                if edit_mode_key not in st.session_state:
                    st.session_state[edit_mode_key] = False

                with st.expander("📖 Upload Document Summary"):
                    st.code(summary)

                # Set edit mode on edit button
                if st.button("✏️ Edit Classification", key=edit_button_key):
                    st.session_state[edit_mode_key] = True

                # Render based on session state
                if st.session_state[edit_mode_key]:
                    corrected_dan, guidance_keywords = render_editable_form(i, top, retention_df, dan_lookup)

                    if st.button("✅ Confirm Update", key=confirm_button_key):
                        # Save logic...
                        was_correct = corrected_dan == top["dan"]
                        title = summary.split("\n")[0] if summary else uploaded_file.name
                        feedback_log_path = "dan_feedback_log.csv"
                        log_exists = os.path.exists(feedback_log_path)

                        try:
                            selected_row = dan_lookup[corrected_dan]
                            feedback_log_path = "dan_feedback_log.csv"
                            log_exists = os.path.exists(feedback_log_path)

                            with open(feedback_log_path, "a", newline="", encoding="utf-8") as f:
                                writer = csv.DictWriter(f, fieldnames=[
                                    "timestamp", "document_name", "title", "predicted_dan", "user_dan",
                                        "was_correct", "match_score", "keywords", "user_keywords", "summary_text",
                                        "description", "retention_period", "designation"
                                ])
                                if not log_exists:
                                    writer.writeheader()
                                writer.writerow({
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "document_name": uploaded_file.name,
                                    "title": title,
                                    "predicted_dan": top["dan"],
                                    "user_dan": corrected_dan,
                                    "was_correct": was_correct,
                                    "match_score": round(top["match_score"], 4),
                                    "keywords": extract_keywords(text, user_keywords=guidance_keywords),
                                    "user_keywords": guidance_keywords.strip(),
                                    "summary_text": summary.replace("\n", " "),
                                    "description": selected_row.get("description", ""),
                                    "retention_period": selected_row.get("retention_period", ""),
                                    "designation": selected_row.get("designation", "")
                                })
                            st.success("✅ Feedback saved.")
                        except Exception as e:
                            st.error(f"Failed to write feedback: {e}")

                        # Switch to read-only view after confirming
                        st.session_state[edit_mode_key] = False
                        # Show read-only view explicitly after confirm
                        #render_read_only_view(top, matches)
                        st.rerun()

                else:
                    render_read_only_view(top, matches)

else:
    st.warning("⚠️ No retention schedule loaded. Upload a file or enable reuse above.")
