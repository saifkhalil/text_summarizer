"""Text extraction from uploaded PDF/DOCX files."""
from __future__ import annotations

import logging

import pdfplumber
from docx import Document

logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: str) -> str:
    parts: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
    except Exception:
        logger.exception("Failed to extract text from PDF: %s", path)
        return ""
    return "\n".join(parts).strip()


def extract_text_from_docx(path: str) -> str:
    try:
        doc = Document(path)
    except Exception:
        logger.exception("Failed to open DOCX: %s", path)
        return ""
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras).strip()
