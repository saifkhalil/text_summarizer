import os, uuid
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from file_extractors import extract_text_from_pdf, extract_text_from_docx
from summarizer import summarize_text
from i18n import t

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Text Summarization System (Ollama v6)")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

def word_count(s: str) -> int:
    return len((s or "").split())

def get_ratio(length: str) -> float:
    return {"short":0.15, "medium":0.25, "long":0.40}.get(length, 0.25)

def safe_ui_lang(lang: str) -> str:
    return lang if lang in ("en","ar") else "en"

@app.get("/", response_class=HTMLResponse)
def index(request: Request, ui_lang: str="en"):
    ui_lang = safe_ui_lang(ui_lang)
    return templates.TemplateResponse("index.html", {"request":request, "ui_lang":ui_lang, "t":lambda k: t(ui_lang,k)})

@app.post("/summarize", response_class=HTMLResponse)
async def summarize(
    request: Request,
    ui_lang: str = Form(default="en"),
    content_lang: str = Form(default="auto"),
    engine: str = Form(default="ollama_local"),
    mode: str = Form(default="fallback"),
    style: str = Form(default="both"),
    ollama_url: str = Form(default=os.getenv("OLLAMA_URL","http://localhost:11434")),
    model: str = Form(default=os.getenv("OLLAMA_MODEL","qwen2.5:7b-instruct")),
    ollama_api_key: str = Form(default=os.getenv("OLLAMA_API_KEY","")),
    input_text: str = Form(default=""),
    length: str = Form(default="medium"),
    file: UploadFile = File(default=None),
):
    ui_lang = safe_ui_lang(ui_lang)
    content_lang = content_lang if content_lang in ("auto","en","ar") else "auto"
    engine = engine if engine in ("ollama","extractive") else "ollama"
    mode = mode if mode in ("strict","fallback") else "fallback"
    style = style if style in ("paragraph","bullets","both") else "both"

    text = (input_text or "").strip()

    if file and file.filename:
        ext = os.path.splitext(file.filename.lower())[1]
        path = os.path.join(UPLOAD_DIR, str(uuid.uuid4()) + ext)
        with open(path, "wb") as f:
            f.write(await file.read())
        if ext == ".pdf":
            text = extract_text_from_pdf(path)
        elif ext == ".docx":
            text = extract_text_from_docx(path)

    if not text:
        return templates.TemplateResponse("result.html", {
            "request":request,"ui_lang":ui_lang,"t":lambda k: t(ui_lang,k),
            "error": t(ui_lang,"no_text"),"engine_error":"","original":"","summary":"",
            "wc_original":0,"wc_summary":0,"detected_lang":"","content_lang":content_lang,
            "length":length,"engine":engine,"used_engine":"","mode":mode,"style":style,
            "ollama_url":ollama_url,"model":model
        })

    summary, detected_lang, used_engine, engine_error = summarize_text(
        text, ratio=get_ratio(length), length=length, content_lang=content_lang,
        engine=engine, mode=mode, style=style, ollama_url=ollama_url, model=model, ollama_api_key=ollama_api_key
    )

    if engine.startswith("ollama") and mode == "strict" and not summary:
        return templates.TemplateResponse("result.html", {
            "request":request,"ui_lang":ui_lang,"t":lambda k: t(ui_lang,k),
            "error": t(ui_lang,"error_ollama"),"engine_error":engine_error,"original":text,"summary":"",
            "wc_original":word_count(text),"wc_summary":0,"detected_lang":detected_lang,"content_lang":content_lang,
            "length":length,"engine":engine,"used_engine":used_engine,"mode":mode,"style":style,
            "ollama_url":ollama_url,"model":model
        })

    return templates.TemplateResponse("result.html", {
        "request":request,"ui_lang":ui_lang,"t":lambda k: t(ui_lang,k),
        "error":"", "engine_error": engine_error, "original":text, "summary":summary,
        "wc_original":word_count(text),"wc_summary":word_count(summary),
        "detected_lang":detected_lang,"content_lang":content_lang,
        "length":length,"engine":engine,"used_engine":used_engine,"mode":mode,"style":style,
        "ollama_url":ollama_url,"model":model
    })
