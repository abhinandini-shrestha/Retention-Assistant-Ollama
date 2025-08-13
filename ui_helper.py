import streamlit as st
import pandas as pd

def format_description_sm(text: str) -> str:
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

def add_footer_note():
   """
   Displays a fixed footer at the bottom of the page
   crediting the University of Washington students.
   """

   footer=f"""
        <style>
           .footer {{
               position: fixed;
               left: 0;
               bottom: 0;
               width: 100%;
               background-color: #4B2E83;
               color: white;
               text-align: right;
               padding: 0.5rem;
               font-size: 0.85rem;
               z-index: 999992;
               padding-right: 20px;
           }}
           .footer a {{
                       color: #b7a57a; /* UW gold */
                       text-decoration: none;
           }}
           .footer p {{
                right: 10px
           }}
       </style>
       <div class="footer">
           Developed by : Students of <a href="https://ischool.uw.edu/" target="_blank">&nbsp;University of Washington</a>
           &nbsp;|&nbsp; <a href="https://github.com/abhinandini-shrestha/Retention-Assistant-Ollama" target="_blank">GitHub Repo</a>
       </div>
   """
   st.markdown(footer,unsafe_allow_html=True)
