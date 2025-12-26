import os
import uuid
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from file_extractors import extract_text_from_pdf, extract_text_from_docx
from summarizer import summarize_text
from i18n import t

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Text Summarization System (Multilang UI)")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

def word_count(s: str) -> int:
    return len((s or "").split())

def get_ratio(length: str) -> float:
    return {"short": 0.15, "medium": 0.25, "long": 0.40}.get(length, 0.25)

def safe_lang(lang: str) -> str:
    return lang if lang in ("en", "ar") else "en"

@app.get("/", response_class=HTMLResponse)
def index(request: Request, ui_lang: str = "en"):
    ui_lang = safe_lang(ui_lang)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "ui_lang": ui_lang, "t": lambda k: t(ui_lang, k)},
    )

@app.post("/summarize", response_class=HTMLResponse)
async def summarize(
    request: Request,
    ui_lang: str = Form(default="en"),
    content_lang: str = Form(default="auto"),  # auto | en | ar
    input_text: str = Form(default=""),
    length: str = Form(default="medium"),
    file: UploadFile = File(default=None),
):
    ui_lang = safe_lang(ui_lang)
    content_lang = content_lang if content_lang in ("auto", "en", "ar") else "auto"

    extracted_text = ""
    if input_text and input_text.strip():
        extracted_text = input_text.strip()

    if file is not None and file.filename:
        ext = os.path.splitext(file.filename.lower())[1]
        file_id = str(uuid.uuid4())
        saved_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")

        content = await file.read()
        with open(saved_path, "wb") as f:
            f.write(content)

        if ext == ".pdf":
            extracted_text = extract_text_from_pdf(saved_path)
        elif ext == ".docx":
            extracted_text = extract_text_from_docx(saved_path)
        else:
            extracted_text = ""

    if not extracted_text:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "ui_lang": ui_lang,
                "t": lambda k: t(ui_lang, k),
                "error": t(ui_lang, "no_text"),
                "original": "",
                "summary": "",
                "wc_original": 0,
                "wc_summary": 0,
                "detected_lang": "",
            },
        )

    ratio = get_ratio(length)
    summary, detected_lang = summarize_text(extracted_text, ratio=ratio, content_lang=content_lang)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "ui_lang": ui_lang,
            "t": lambda k: t(ui_lang, k),
            "error": "",
            "original": extracted_text,
            "summary": summary,
            "wc_original": word_count(extracted_text),
            "wc_summary": word_count(summary),
            "detected_lang": detected_lang,
            "content_lang": content_lang,
            "length": length,
        },
    )

@app.get("/set-lang")
def set_lang(ui_lang: str = "en"):
    # convenience redirect: /set-lang?ui_lang=ar
    ui_lang = safe_lang(ui_lang)
    return RedirectResponse(url=f"/?ui_lang={ui_lang}")
