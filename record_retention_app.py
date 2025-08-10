import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" #Hugging face transformer library
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from datetime import datetime

from csv_utils import (
    generate_csv_if_missing, 
    clear_retention_schedules, 
    get_document_uid, 
    upsert_feedback_row, 
    append_user_keywords
)
from retention_utils import (
    build_keyword_stats,
    match_retention,
    extract_text,
    summarize_without_nlp,
    clear_popup,
    dedupe_batch,
    #summarize_with_nlp,
    #load_spacy_model
    )
from ui_helper import format_description_sm



FEEDBACK_CSV_PATH = "dan_feedback_log.csv"

st.set_page_config(page_title="🏛️ DOL - Retention Assistant")

st.warning(
    """
    **Notice:** This tool is designed to assist in understanding and classifying documents according to the retention schedule.
    It does not replace official review. If you are uncertain about a label, retention period,
    or archival requirements, please consult your supervisor before finalizing classification.
    """
)

st.title("🏛️ DOL - Retention Assistant")
st.sidebar.markdown(" ## 🧭 Quick Access ")
st.sidebar.markdown("↑ [Top](#dol-retention-assistant)")



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

# ☑️ Use previous schedule toggle
retention_df = None
all_rules = []
new_upload = False
retention_files = []

#def _collect_keywords_from_sources - Removed

use_previous_schedule = st.checkbox("Use previous retention schedule", value=True)
csv_files = [f for f in os.listdir(retention_dir) if f.endswith(".csv")]

# If user opt not to select previous schedule or there is not csv
if not use_previous_schedule or len(csv_files) == 0:
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
                    df.to_csv(csv_path, index=False)  # update CSV with metadata
                    all_rules.append(df)

        if all_rules:
            retention_df = pd.concat(all_rules, ignore_index=True)
            st.success(f"✅ Loaded {len(retention_df)} categories from {file.name}.")

# If using previous and a CSV is previously uploaded
else:
    all_dfs = []
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
                    <div style="
                        background-color: var(--secondary-background-color);
                        color: var(--text-color);
                        padding: 1rem;
                        border-left: 5px solid #1e90ff;
                        border-radius: 0.5rem;
                        margin-bottom: 1rem;
                    ">
                        <b>Source PDF:</b> {source}<br>
                        <b>Uploaded on:</b> {date}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:
            popup_messages = st.warning(f"⚠️ Could not read `{f}`: {e}")

        if all_dfs:
                retention_df = pd.concat(all_dfs, ignore_index=True)


# Show preview if we have a DataFrame
if retention_df is not None and not retention_df.empty:

    keyword_stats = build_keyword_stats(retention_df)

    # Build DAN lookup dictionary once
    dan_lookup = retention_df.drop_duplicates(subset="dan").set_index("dan").to_dict("index")
    dan_options = list(dan_lookup.keys())

    displayed_doc_uids = set()
    seen_doc_uids = st.session_state.setdefault("seen_doc_uids", set()) #to see if the user is trying to upload same document twice

    uploaded_files = st.file_uploader("📄 Upload Documents to Classify", type=["pdf", "docx", "txt", "png", "jpg"], accept_multiple_files=True)
    if uploaded_files:
        dup_names = []

        for i, uploaded_file in enumerate(uploaded_files):
            doc_uid = get_document_uid(uploaded_file)

            if doc_uid in displayed_doc_uids:
                dup_names.append(uploaded_file.name)
                st.warning(f"⚠️ Duplicate in this batch skipped: **{uploaded_file.name}**")
                continue

            displayed_doc_uids.add(doc_uid)

            # (Optional) still let the user know if they re-uploaded something seen before
            first_time_this_session = doc_uid not in seen_doc_uids
            seen_doc_uids.add(doc_uid)


            #in-page anchor for each uploads
            st.markdown(f'<a name="anchor_{doc_uid}"></a>', unsafe_allow_html=True)
            st.sidebar.markdown(f"📄[{uploaded_file.name}](#anchor_{doc_uid})")


            with st.expander(f"📄 Document: {uploaded_file.name}", expanded=True):


                text = extract_text(uploaded_file)
                if not text.strip():
                    st.warning("📄 No readable text found.")
                    continue

                #system predicted matches
                matches = match_retention(text, retention_df, model, keyword_stats)
                selected = matches[0]
                predicted_dan = selected["dan"]

                #action when user inputs keywords
                context_input = st.text_input("📝 Optional: Add context keywords (comma-separated)", key=f"context_{doc_uid}")
                keywords = [kw.strip().lower() for kw in context_input.split(",")] if context_input else []

                if not text.strip() and not context_input.strip():
                    popup_messages = st.warning("⚠️ No readable text or context provided.")
                    continue

                combined_text = f"{context_input.strip()} {text.strip()}"
                keyword_stats = {kw: 3.0 for kw in context_input.strip()}

                #match after user inputs keywords
                matches = match_retention(combined_text, retention_df, model, keyword_stats=None)
                selected = matches[0]
                user_dan = selected["dan"]

                st.markdown("🥇Top 3 Possible Matches")
                # Show each match in its own expander
                for idx, match in enumerate(matches, start=1):
                    star = "⭐ " if idx == 1 else ""
                    with st.expander(f"[{match['dan']}] {match['dan_title']}", expanded=(idx == 1)):
                        st.markdown(format_description_sm(f"<b>{match['dan_title']}</b><br>{match['dan_description']}"), unsafe_allow_html=True)
                        st.markdown(f"**Retention Period:** {match['dan_retention']}")
                        st.markdown(f"**Designation:** {match['dan_designation']}")
                        st.markdown(f"**Source:** {match['source_pdf']}")


                confirm_match = st.button("Confirm Match", width=150, type="primary", key=f"confirm_{i}" )


                doc_name = ((uploaded_file.name if 'uploaded_file' in locals() and uploaded_file is not None else None) or "")
                # Write feedback row EXACTLY with the schema
                row = {
                    "document_uid": doc_uid,  
                    "timestamp": datetime.utcnow().isoformat(),
                    "document_name": doc_name,
                    "predicted_dan": predicted_dan,
                    "user_dan": user_dan,
                    "was_correct": predicted_dan == user_dan,
                    "match_score": selected["match_score"],
                    "user_keywords": ", ".join(keywords),
                    "dan_title": selected["dan_title"],
                    "dan_category": selected["dan_category"],
                    "dan_retention": selected["dan_retention"],
                    "dan_designation": selected.get("dan_designation", ""),
                    "summary_text": summarize_without_nlp(text),
                    "source_pdf": selected.get("source_pdf", ""),

                }

                if confirm_match:
                    # try to infer doc name/title from your existing variables; fall back safely                    
                    upsert_feedback_row(row)
                    popup_messages = st.success("Confirmed DAN!")
                    clear_popup(popup_messages)
            

                st.info("💡 Help improve the system’s accuracy by adding related keywords.")

                feedback_input = st.text_input("🏷️ Add related keywords (comma-separated) to improve this DAN match", placeholder="Example: Business and professional licenses, Complaints and grievances, ...", key=f"feedback_{doc_uid}")

                if feedback_input:
                    append_user_keywords(doc_uid, feedback_input, row)
                    popup_messages = st.success("✅ Keywords appended")
                    clear_popup(popup_messages)


# Export feedback_csv
if os.path.exists(FEEDBACK_CSV_PATH):
    df = pd.read_csv(FEEDBACK_CSV_PATH)
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button(
        label="⬇️ Download Logs",
        data=csv_bytes,
        file_name="feedback_log.csv",
        mime="text/csv"
    )
else:
    st.sidebar.info("No feedback log available.")
                    

