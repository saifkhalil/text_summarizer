import pdfplumber
from docx import Document
import fitz

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    return "\n".join(parts).strip()

def extract_text_from_docx(path: str) -> str:
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras).strip()
