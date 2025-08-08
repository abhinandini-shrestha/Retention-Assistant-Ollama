import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import tempfile
import pandas as pd
import streamlit as st
from sentence_transformers import util, SentenceTransformer
from generate_dan_csv import generate_csv_if_missing, extract_keywords, list_uploaded_sources, clear_retention_schedules
from ui_helper import format_description_md, render_read_only_view, render_editable_form
import csv
from datetime import datetime

from retention_utils import (
    normalize_text,
    build_keyword_stats,
    match_retention,
    extract_text,
    #summarize_with_nlp,
    summarize_without_nlp,
    #load_spacy_model
    )

st.set_page_config(page_title="📁 Washington Records Retention Assistant")
st.title("📁 Washington Records Retention Assistant")

# Load models
#@st.cache_resource
#def get_nlp():
    #return load_spacy_model()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

#nlp = get_nlp()
model = load_embedding_model()

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
FEEDBACK_CSV_PATH = "dan_feedback_log.csv"



def _ensure_feedback_header(path: str):
    cols = [
        "timestamp","document_name","title","predicted_dan","user_dan","was_correct",
        "match_score","keywords","user_keywords","summary_text",
        "category_description","retention_period","designation"
        # "schedule_source"  # intentionally not used right now
    ]
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()

def _append_feedback_row(row: dict, path: str = FEEDBACK_CSV_PATH):
    cols = [
        "timestamp","document_name","title","predicted_dan","user_dan","was_correct",
        "match_score","keywords","user_keywords","summary_text",
        "category_description","retention_period","designation"
    ]
    _ensure_feedback_header(path)
    # ensure all keys exist
    clean = {c: row.get(c, "") for c in cols}
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=cols).writerow(clean)

def _collect_keywords_from_sources(retention_df, feedback_csv_path: str) -> list[str]:
    """We only USE this to help the user fill the one text box; UI stays minimal like the PDF."""
    suggestions = set()
    # feedback: keywords, user_keywords
    try:
        if os.path.exists(feedback_csv_path):
            fdf = pd.read_csv(feedback_csv_path)
            for col in ["keywords","user_keywords"]:
                if col in fdf.columns:
                    fdf[col] = fdf[col].fillna("")
                    for cell in fdf[col].astype(str).tolist():
                        for tok in re.split(r"[;,]\\s*|\\s*,\\s*", cell):
                            tok = tok.strip()
                            if tok:
                                suggestions.add(tok)
    except Exception:
        pass
    # retention: Keywords / Description-ish columns
    if retention_df is not None and not retention_df.empty:
        for col in ["Keywords","keywords","Description of Records","DESCRIPTION OF RECORDS","Category Description","category_description","Description","DESCRIPTION"]:
            if col in retention_df.columns:
                for cell in retention_df[col].fillna("").astype(str).tolist():
                    parts = re.split(r"[;,,]\\s*", cell)
                    if any(p.strip() for p in parts if p):
                        for p in parts:
                            p = p.strip()
                            if p:
                                suggestions.add(p)
    return sorted(suggestions, key=str.lower)

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


            # Build top 3 from whatever your matcher produced.
            # Expecting a list[dict] 'matches' with keys: dan, retention, description, designation, score, title (if available)
        _top3_records = []
#             if "matches" in locals() and isinstance(matches, list) and len(matches):
#                 for m in matches[:3]:
#                     _top3_records.append({
#                         "DAN": m.get("dan",""),
#                         "Retention": m.get("retention",""),
#                         "_desc": m.get("description",""),
#                         "_designation": m.get("designation",""),
#                         "_score": m.get("score",""),
#                         "_title": m.get("title","")
#                     })
#             else:
#                 # empty state that still matches your PDF’s simple table
#                 _top3_records = [
#                     {"DAN": "", "Retention": "", "_desc": "", "_designation": "", "_score": "", "_title": ""},
#                     {"DAN": "", "Retention": "", "_desc": "", "_designation": "", "_score": "", "_title": ""},
#                     {"DAN": "", "Retention": "", "_desc": "", "_designation": "", "_score": "", "_title": ""},
#                 ]

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

                summary = summarize_without_nlp(text)
                _matches = match_retention(text, retention_df, model, keyword_stats)
                _selected = _matches[0]

    #             _top3_df = pd.DataFrame(_top3_records)[["DAN","Retention"]]
    #             st.dataframe(_top3_df, use_container_width=True, hide_index=True)


                context_input = st.text_input("📝 Optional: Add context keywords (comma-separated)...", key=f"context_{i}")
                keywords = [kw.strip().lower() for kw in context_input.split(",")] if context_input else []

                if not text.strip() and not context_input.strip():
                    st.warning("⚠️ No readable text or context provided.")
                    continue

                combined_text = f"{context_input.strip()} {text.strip()}"
                keyword_stats = {kw: 3.0 for kw in context_input.strip()}
                _matches = match_retention(combined_text, retention_df, model, keyword_stats=None)
                _selected = _matches[0]

                st.markdown("🥇Top 3 Possible Matches")
                top_matches_df = pd.DataFrame([
                        {
                            "DAN": m["dan"],
                            "Retention": m["retention_period"],
                        }
                        for m in _matches[0:3]
                    ])

                st.table(top_matches_df)

                # Readouts block (plain text like the PDF)
                st.markdown(f"**DAN**: `{_selected['dan']}`")
                st.markdown(format_description_md(_selected["description"]))
                st.markdown(f"**Retention**: {_selected['retention_period']}")
                st.markdown(f"**Designation**: {_selected['designation']}")


                c1, c2 = st.columns([1,1])
                with c1:
                    _confirm = st.button("Confirm", type="primary", use_container_width=True, key=f"confirm_{i}" )
                #with c2:
                    #st.button("Override DAN", disabled=True, use_container_width=True, help="Disabled for now", key="pdf_like_override")

                if _confirm:
                    # try to infer doc name/title from your existing variables; fall back safely
                    _doc_name = (
                        (uploaded_file.name if 'uploaded_file' in locals() and uploaded_file is not None else None)
                        or (document_name if 'document_name' in locals() else None)
                        or ""
                    )
                    _title = _selected.get("_title","") or (_selected.get("_desc","")[:80] if _selected.get("_desc") else "")

                    # Write feedback row EXACTLY with your schema
                    _append_feedback_row({
                        "timestamp": datetime.utcnow().isoformat(),
                        "document_name": _doc_name,
                        "title": _title,
                        "predicted_dan": _selected["dan"],
                        "user_dan": "",
                        "was_correct": "",
                        "match_score": _selected["match_score"],
                        "keywords": "",
                        "user_keywords": "",
                        "summary_text": "",
                        "category_description": _selected["description"],
                        "retention_period": _selected["retention_period"],
                        "designation": _selected.get("designation", "")
                    }, FEEDBACK_CSV_PATH)
                    st.success("Saved to feedback log.")


                # ==== END: PDF-lookalike UI ====
