# 🗃️ Washington Records Retention Assistant

A local, privacy-friendly app to help match public records to the appropriate retention schedules. It uses semantic similarity (Sentence-Transformer) and a human-in-the-loop feedback system to recommend the most likely DANs (Disposition Authority Numbers).

## ⚙️ Installation
1. Clone the repository
```bash
git clone https://github.com/abhinandini-shrestha/Retention-Assistant.git
cd retention-assistant
```

2. Create a virtual environment $ install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Running the App

```bash
streamlit run record_retention_app.py
```

## 🔍 Features

- Upload and parse WA state retention schedules (PDFs). Performs PDF to CSV conversion. 
- Upload one or more documents (PDF, DOCX, TXT, JPG, PNG)
- Automatically match document content to retention rules using local embeddings (MiniLM)
- Top-3 DAN Match Suggestions
- OCR support for scanned PDFs and embedded Word images
- Human-in-the-loop Feedback
    - Confirm or correct predicted DAN
    - Add supporting keywords
    - Automatically logs corrections to dan_feedback_log.csv
- Versioned Schedules: stores uploaded schedules with source_pdf and upload_date metadata in csv files
- Audit Logging: Feedback CSV includes timestamp, document name, predicted DAN, corrected DAN, match score, overlapping keywords, and summary text
- Sidebar Navigation for quick access
- Export Feedback CSV for compliance reporting

## 🔎 Approach & Methodology

We took this approach to reduce the time required to look up DANs, as employees currently still need to manually search and match them through the retention schedule.

- Data Preparation: Retention schedule PDFs are transformed into structured CSV files containing essential fields such as DAN, Title, Category Description, and Retention Period.
- Data Cleaning: Removes unnecessary keywords or irrelevant sections of documents to boost classification speed and efficiency.
- Document Processing: Uploaded documents (PDF, Word, or image formats) are parsed to extract text. Optical Character Recognition (OCR) is applied to image-based files.
- Matching Algorithm: Classification primarily relies on semantic similarity between document text and retention schedule entries. Keyword search and user feedback logs are also incorporated to improve accuracy.
- Feedback Logs: Users confirm or correct the predicted DAN. Corrections are stored in dan_feedback_log.csv, allowing the system to learn and improve over time.

## 🛠️ Requirements
Main dependencies used in this project:

- streamlit – UI framework
- sentence-transformers – semantic similarity model
- torch, transformers – required for embeddings
- pandas, numpy – data processing
- pdfplumber – PDF text extraction
- pytesseract – OCR for image-based documents
- Pillow – image handling

(All dependencies are listed in requirements.txt)

## 📂 Project Structure
```bash
📦 retention-assistant
├── record_retention_app.py     # Main Streamlit app
├── match_decision_process.py   # Matching logic (semantic + keyword rules + feedback)
├── retention_util.py           # Helper class
├── csv_utils.py                # CSV-related tasks (PDF → CSV, feedback log updates
├── retentions_schedule/
├── ├── schedules/              # Uploaded retention schedule CSVs
├── dan_feedback_log.csv        # User feedback log
└── README.md                   # Project documentation
```

## 📁 Output

- Classifications and manual edits saved to `dan_feedback_log.csv`
- Downloadable via the app UI

---
