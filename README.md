# 🗃️ Washington Records Retention Assistant Ollama

A local, privacy-friendly app to help match public records to the appropriate retention schedules using semantic search and OCR.

## 🔍 Features

- Upload and parse WA state retention schedules (PDFs)
- Upload one or more documents (PDF, DOCX, TXT, JPG, PNG)
- Automatically match document content to retention rules using local embeddings (MiniLM)
- Confidence scoring (High / Medium / Low)
- OCR support for scanned PDFs and embedded Word images
- Manual "Edit" or "Confirm" classification
- Download full CSV of matched results

## 🛠️ Requirements

- Python 3.9+
- Tesseract OCR
- Ollama (optional for LLM-based prompting – currently not active)
- pip packages listed in `requirements.txt`

## 🚀 Running the App

```bash
cd retention-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run record_retention_app.py
```

## 📦 Deployment Notes

This app runs entirely locally. If Tesseract is not installed, install it via:

### macOS
```bash
brew install tesseract
```

### Ubuntu
```bash
sudo apt install tesseract-ocr
```


## 📁 Output

- Classifications and manual edits saved to `results_log.csv`
- Downloadable via the app UI

---
