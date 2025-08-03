import os
import re
import io
import zipfile
import tempfile
import pytesseract
import docx2txt
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
from collections import Counter
from sentence_transformers import util

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def build_keyword_stats(rules_df):
    all_words = []
    for desc in rules_df['description']:
        words = re.findall(r'\b\w+\b', desc.lower())
        all_words.extend(words)
    return Counter(all_words)

def match_retention_priority(text, schedule_df, model, keyword_stats, top_n=3):
    clean_text = normalize_text(text)
    doc_words = set(clean_text.split())
    doc_embedding = model.encode([clean_text], convert_to_tensor=True)
    scored_matches = []
    for _, row in schedule_df.iterrows():
        desc = normalize_text(row["description"])
        desc_words = set(desc.split())
        overlap = doc_words & desc_words
        #rarity_score = sum(1 / (keyword_stats[w] + 1) for w in overlap)
        entry_embedding = model.encode([desc], convert_to_tensor=True)
        final_score = float(util.cos_sim(doc_embedding, entry_embedding)[0][0]) #usually between 0.3–0.9
        #final_score = 0.6 * semantic_score + 0.4 * rarity_score
        scored_matches.append({**row, "match_score": round(final_score, 4)})
    return sorted(scored_matches, key=lambda x: x["match_score"], reverse=True)[:top_n]

def extract_text_from_pdf_images(pdf_path):
    doc = fitz.open(pdf_path)
    all_text = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image = Image.open(io.BytesIO(base_image["image"])).convert("L")
            image = image.filter(ImageFilter.MedianFilter())
            image = ImageEnhance.Contrast(image).enhance(2.0)
            text = pytesseract.image_to_string(image, config="--psm 6")
            if text.strip():
                all_text.append(text.strip())
    return "\n\n".join(all_text)

def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        text = ""
        file_bytes = uploaded_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(file_bytes)
            tmpfile_path = tmpfile.name

        with fitz.open(tmpfile_path) as doc:
            for page in doc:
                page_text = page.get_text("text").strip()
                if page_text:
                    text += page_text + "\n"

        ocr_text = extract_text_from_pdf_images(tmpfile_path)
        if ocr_text:
            text += "\n" + ocr_text

        os.remove(tmpfile_path)
        return text

    elif uploaded_file.name.endswith(".docx"):
        text = docx2txt.process(uploaded_file)
        if text.strip():
            return text
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.read())
            with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
                image_texts = []
                for file in zip_ref.namelist():
                    if file.startswith("word/media/") and file.endswith((".png", ".jpg", ".jpeg")):
                        zip_ref.extract(file, tmpdir)
                        img = Image.open(os.path.join(tmpdir, file))
                        image_texts.append(pytesseract.image_to_string(img))
                return "\n".join(image_texts)

    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")

    elif uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        return pytesseract.image_to_string(image)

    return ""

def summarize_with_nlp(text, nlp, max_sents=3):
    doc = nlp(text)
    return "\n".join(str(s) for s in list(doc.sents)[:max_sents])

