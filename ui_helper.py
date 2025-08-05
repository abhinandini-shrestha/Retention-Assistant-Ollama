import streamlit as st
import pandas as pd

def format_description_md(text: str) -> str:
    lines = text.splitlines()
    formatted_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("•"):
            # Convert bullet to markdown list
            formatted_lines.append(f"- {stripped[1:].strip()}")
        else:
            formatted_lines.append(stripped)
    return "\n".join(formatted_lines)

def render_read_only_view(top, matches):
    st.markdown(f"**🥇 Top Possible Match**:")
    st.markdown(f"**DAN**: `{top['dan']}`")
    st.markdown(format_description_md(top["description"]))
    st.markdown(f"**Retention**: {top['retention_period']}")
    st.markdown(f"**Designation**: {top['designation']}")
    #st.markdown(f"**Confidence Score:** {top['match_score'] * 100:.0f}%")
    st.markdown("**Other Relevent Matches:**")
    top_matches_df = pd.DataFrame([
        {
            "DAN": m["dan"],
            "Retention": m["retention_period"],
            "Score (%)": f"{m['match_score'] * 100:.1f}%"
        }
        for m in matches[1:3]
    ])

    st.table(top_matches_df)



def render_editable_form(i, top, retention_df, dan_lookup):
    selectbox_key = f"select_dan_{i}"
    keywords_key = f"keywords_{i}"

    dan_options = list(dan_lookup.keys())
    corrected_dan = st.selectbox("Select DAN:", dan_options, index=dan_options.index(top["dan"]), key=selectbox_key)
    selected_row = dan_lookup[corrected_dan]

    st.markdown(format_description_md(selected_row["description"]))
    st.markdown(f"**Retention:** {selected_row['retention_period']}")
    st.markdown(f"**Designation:** {selected_row['designation']}")

    guidance_keywords = st.text_area("🧠 Keywords that helped you decide:", key=keywords_key)

    return corrected_dan, guidance_keywords