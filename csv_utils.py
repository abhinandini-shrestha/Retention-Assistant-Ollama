import os
import hashlib, re
import csv
import tempfile
import pdfplumber
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from retention_utils import extract_keywords
from pathlib import Path
from typing import Optional
import math

RETENTION_DIR = "retention_schedules"
FEEDBACK_CSV_PATH = "dan_feedback_log.csv"
FEEDBACK_COLUMNS = [
    "document_uid", "timestamp","document_name","predicted_dan","user_dan","was_correct",
    "match_score","user_keywords","dan_title","dan_category","dan_retention",
    "dan_designation","summary_text", "schedule_source"
]

def sanitize(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val)

##### FOR RETENTION SCHEDULE CSV #######

def extract_retention_table(pdf_file):
    rules = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                category = ''
                if table[0][0]:
                    category = table[0][0].split('\n')[0].strip()

                header_row_index = -1
                for i, row in enumerate(table):
                    normalized = [str(cell).upper().strip() if cell else "" for cell in row]
                    match_count = sum(1 for keyword in ["AUTHORITY", "DESCRIPTION", "RETENTION", "DESIGNATION"]
                                      if any(keyword in cell for cell in normalized))
                    if match_count >= 2:
                        header_row_index = i
                        break
                if header_row_index == -1:
                    continue

                for row in table[header_row_index + 1:]:
                    if not row or len(row) < 2:
                        continue
                    
                    # Normalize row to ensure at least 6 elements
                    row = [cell.strip() if cell else "" for cell in row]
    
                    row += [""] * (6 - len(row))  # pad with empty strings
                    dan = row[0]

                    # Pick the first non-empty value among row[1], row[2], row[3]
                    desc_index = next((i for i in (1, 2, 3, 4) if row[i].strip()), None)

                    if desc_index is not None:
                        full_description = row[desc_index].strip()
                        retention = row[desc_index + 1] if desc_index + 1 < len(row) else ""
                        designation = row[desc_index + 2] if desc_index + 2 < len(row) else ""
                    else:
                        full_description = ""
                        retention = ""
                        designation = ""

                    if not dan or not full_description:
                        continue

                    title, description = split_title_and_description(full_description)
                        
                    keywords = extract_keywords(full_description)
                    rules.append({
                        "dan": dan,
                        "dan_title": title,
                        "dan_description": description,
                        "dan_retention": retention,
                        "dan_designation": designation,
                        "dan_category": category,
                        "keywords": keywords
                    })

    return rules

def generate_csv_if_missing(pdf_file, use_previous=True):
    """
    Converts uploaded PDF into a CSV file.
    Deletes old files if use_previous is False.
    Returns path to the new CSV.
    """
    os.makedirs(RETENTION_DIR, exist_ok=True)

    # Create safe and unique file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = pdf_file.name.replace(" ", "_").replace(".pdf", "")
    csv_filename = f"{timestamp}__{base_name}.csv"
    csv_path = os.path.join(RETENTION_DIR, csv_filename)

    # Save uploaded PDF to temp file for parsing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    rules = extract_retention_table(tmp_path)
    os.remove(tmp_path)

    if rules:
        df = pd.DataFrame(rules)
        df['source_pdf'] = pdf_file.name
        df['upload_date'] = datetime.now().isoformat()
        df.to_csv(csv_path, index=False)
        return csv_path
    return None

def clear_retention_schedules():
    """
    Deletes all saved schedule CSVs from the RETENTION_DIR folder.
    """
    if os.path.exists(RETENTION_DIR):
        for f in os.listdir(RETENTION_DIR):
            if f.endswith(".csv"):
                os.remove(os.path.join(RETENTION_DIR, f))

def split_title_and_description(text):
    """
    Splits a text into title and description.
    Just taking the first non-empty line as title and the rest as description.
    """
    if not text:
        return "", ""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    title = lines[0] if lines else ""
    description = "\n".join(lines[1:]) if len(lines) > 1 else ""
    return title, description


##### FOR FEEDBACK CSV #######

def get_document_uid(uploaded_file) -> str:
    # same file content -> same ID (even if the name changes)
    digest = hashlib.sha1(uploaded_file.getvalue()).hexdigest()[:12]
    safe_name = re.sub(r"\W+", "_", uploaded_file.name)
    return f"{digest}_{safe_name}"

def ensure_feedback_header(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS)
            writer.writeheader()

def append_feedback_row(row: dict, path: str = FEEDBACK_CSV_PATH):
    ensure_feedback_header(path)
    # ensure all keys exist
    clean = {c: row.get(c, "") for c in FEEDBACK_COLUMNS}
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FEEDBACK_COLUMNS).writerow(clean)

def upsert_feedback_row(row: dict, csv_path: str = FEEDBACK_CSV_PATH):
    path = Path(csv_path)
    
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=FEEDBACK_COLUMNS)

    # ensure all columns exist
    for c in FEEDBACK_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    # upsert
    uid = row["document_uid"]
    mask = (df["document_uid"] == uid)
    if mask.any():
        # update first match
        idx = df.index[mask][0]
        for k, v in row.items():
            df.at[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row], columns=FEEDBACK_COLUMNS)], ignore_index=True)

    df.to_csv(path, index=False)

def append_user_keywords(document_uid: str, new_phrase: str, row: dict, csv_path: str = FEEDBACK_CSV_PATH) -> None:

    new_phrase = sanitize(new_phrase).strip()
    if not new_phrase:
        return

    path = Path(csv_path)
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=FEEDBACK_COLUMNS)

    # Ensure all expected columns exist
    for c in FEEDBACK_COLUMNS:
        if c not in df.columns:
            df[c] = ""

    mask = (df["document_uid"] == document_uid)

    if not mask.any():
        # Create a fresh row with just uid + keywords + timestamp

        row["user_keywords"] = new_phrase
        upsert_feedback_row(row, csv_path=csv_path)
        return

    idx = df.index[mask][0]
    existing = sanitize(df.at[idx, "user_keywords"]).strip()

    exists = False
    if existing:
        exists = any(k.strip().lower() == new_phrase.lower()
                     for k in existing.split(",") if k.strip())

    if exists:
        return  # no change

    merged = f"{existing}, {new_phrase}" if existing else new_phrase

    # Write the merged keywords (and update timestamp) via your upsert
    upsert_feedback_row(
        {
            "document_uid": document_uid,
            "timestamp": datetime.utcnow().isoformat(),
            "user_keywords": merged,
        },
        csv_path=csv_path
    )


# def list_uploaded_sources(): #REMOVED
