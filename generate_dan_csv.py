import os
import re
import csv
import tempfile
import pdfplumber
from datetime import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def extract_keywords(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return ', '.join(sorted(set(w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 3)))

def extract_retention_table(pdf_file):
    rules = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue

                # ✅ 1. Extract category from first row of the table
                category = ''
                if table[0][0]:
                    category = table[0][0].split('\n')[0].strip()

                # ✅ 2. Find actual header row later in table
                header_row_index = -1
                for i, row in enumerate(table):
                    normalized = [str(cell).upper().strip() if cell else "" for cell in row]
                    match_count = sum(1 for keyword in ["AUTHORITY", "DESCRIPTION", "RETENTION", "DESIGNATION"]
                                      if any(keyword in cell for cell in normalized))
                    if match_count >= 3:
                        header_row_index = i
                        break
                if header_row_index == -1:
                    continue  # skip if no header row found

                # ✅ 3. Parse data rows below header
                for row in table[header_row_index + 1:]:
                    if not row or len(row) < 2:
                        continue
                    row = [cell.strip() if cell else "" for cell in row]
                    dan = row[0]
                    description = row[1]
                    retention = row[2] if len(row) > 2 else ""
                    designation = row[3] if len(row) > 3 else ""
                    if not dan or not description:
                        continue
                    keywords = extract_keywords(description)
                    rules.append({
                        "dan": dan,
                        "description": description,
                        "retention_period": retention,
                        "designation": designation,
                        "keywords": keywords,
                        "category": category
                    })

    return rules

def generate_csv_if_missing(pdf_file):
    #base_name = os.path.splitext(pdf_file.name)[0]
    base_name = 'retention_schedule'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "retention_schedules"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{timestamp}_{base_name}.csv")

    if not os.path.exists(csv_path):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        rules = extract_retention_table(tmp_path)
        if rules:
            with open(csv_path, "w", newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["dan", "description", "retention_period", "designation", "keywords", "category"])
                writer.writeheader()
                writer.writerows(rules)

    os.remove(tmp_path)
    return csv_path
