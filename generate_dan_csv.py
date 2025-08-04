import os
import re
import csv
import tempfile
import pdfplumber
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from retention_utils import extract_keywords

RETENTION_DIR = "retention_schedules"

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

                    if row[1] == "":
                        description = row[3]
                        retention = row[4]
                        designation = row[5]
                    else:
                        description = row[1]
                        retention = row[2]
                        designation = row[3]

                    if not dan or not description:
                        continue
                    keywords = extract_keywords(description)
                    rules.append({
                        "dan": dan,
                        "description": description,
                        "retention_period": retention,
                        "designation": designation,
                        "category": category,
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


def list_uploaded_sources():
    """
    Returns a list of (filename, source_pdf, upload_date) from saved CSV files.
    """
    if not os.path.exists(RETENTION_DIR):
        return []

    rows = []
    for f in os.listdir(RETENTION_DIR):
        if f.endswith(".csv"):
            full_path = os.path.join(RETENTION_DIR, f)
            try:
                df = pd.read_csv(full_path, nrows=1)  # just read metadata row
                if 'source_pdf' in df.columns and 'upload_date' in df.columns:
                    rows.append((f, df['source_pdf'].iloc[0], df['upload_date'].iloc[0]))
            except Exception:
                continue

    return sorted(rows, key=lambda x: x[2], reverse=True)


def clear_retention_schedules():
    """
    Deletes all saved schedule CSVs from the RETENTION_DIR folder.
    """
    if os.path.exists(RETENTION_DIR):
        for f in os.listdir(RETENTION_DIR):
            if f.endswith(".csv"):
                os.remove(os.path.join(RETENTION_DIR, f))
