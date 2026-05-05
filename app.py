"""FastAPI app for the Text Summarization System (Ollama Cloud)."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import bleach
import markdown as md_lib
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fpdf import FPDF

from file_extractors import extract_text_from_docx, extract_text_from_pdf
from i18n import t
from summarizer import summarize_text, summarize_text_stream

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTS = {".pdf", ".docx"}
VALID_LENGTHS = {"short", "medium", "long"}
VALID_STYLES = {"paragraph", "bullets", "both"}

ALLOWED_HTML_TAGS = [
    "p", "br", "hr", "strong", "em", "b", "i", "u", "s", "del",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "blockquote", "code", "pre", "span", "div", "mark",
]
ALLOWED_HTML_ATTRS = {"*": ["class", "style"], "span": ["class", "style"]}


def markdown_to_html(text: str) -> str:
    if not text:
        return ""
    if "<" in text and ">" in text and any(tag in text.lower() for tag in ("<p", "<ul", "<ol", "<h1", "<h2", "<h3", "<strong", "<em")):
        # Already HTML-ish; sanitize and return.
        return bleach.clean(text, tags=ALLOWED_HTML_TAGS, attributes=ALLOWED_HTML_ATTRS, strip=True)
    html = md_lib.markdown(text, extensions=["extra", "sane_lists", "nl2br"])
    return bleach.clean(html, tags=ALLOWED_HTML_TAGS, attributes=ALLOWED_HTML_ATTRS, strip=True)

app = FastAPI(title="Text Summarization System")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def word_count(s: str) -> int:
    return len((s or "").split())


def safe_ui_lang(lang: str) -> str:
    return lang if lang in ("en", "ar") else "en"


def _normalize_choice(value: str, allowed: set[str], default: str) -> str:
    return value if value in allowed else default


def _i18n_for(ui_lang: str) -> dict:
    """Build i18n payload for client-side use."""
    keys = [
        "status_received", "status_extracting", "status_extracted",
        "status_summarizing", "status_merging", "status_done",
        "cancel", "cancelled", "new_summary", "loading",
        "summary", "original_text", "words", "result_title",
        "export_pdf", "back", "error_ollama",
    ]
    return {k: t(ui_lang, k) for k in keys}


def render_result(request: Request, ui_lang: str, *, error: str = "",
                  engine_error: str = "", original: str = "",
                  summary: str = "") -> HTMLResponse:
    return templates.TemplateResponse("result.html", {
        "request": request,
        "ui_lang": ui_lang,
        "t": lambda k: t(ui_lang, k),
        "error": error,
        "engine_error": engine_error,
        "original": original,
        "summary": summary,
        "wc_original": word_count(original),
        "wc_summary": word_count(summary),
    })


async def _save_upload(file: UploadFile) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Save an upload to disk with size limit. Returns (path, ext, error_key)."""
    ext = os.path.splitext((file.filename or "").lower())[1]
    if ext not in ALLOWED_EXTS:
        return None, None, None
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    size = 0
    try:
        with open(path, "wb") as fh:
            while True:
                chunk = await file.read(64 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    fh.close()
                    _safe_remove(path)
                    return None, None, "err_file_too_large"
                fh.write(chunk)
    except Exception:
        logger.exception("Failed saving upload")
        _safe_remove(path)
        return None, None, "no_text"
    return path, ext, None


def _extract_text(path: str, ext: str) -> str:
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext == ".docx":
        return extract_text_from_docx(path)
    return ""


def _safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def _validate_ollama_settings(ui_lang: str) -> Optional[str]:
    if not OLLAMA_URL.strip() or not OLLAMA_MODEL.strip():
        return t(ui_lang, "err_cloud_requires_url_model")
    if not OLLAMA_API_KEY.strip():
        return t(ui_lang, "err_cloud_requires_key")
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request, ui_lang: str = "en") -> HTMLResponse:
    ui_lang = safe_ui_lang(ui_lang)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "ui_lang": ui_lang,
            "t": lambda k: t(ui_lang, k),
            "i18n_json": json.dumps(_i18n_for(ui_lang)),
        },
    )


@app.post("/summarize", response_class=HTMLResponse)
async def summarize(
    request: Request,
    ui_lang: str = Form(default="en"),
    style: str = Form(default="both"),
    input_text: str = Form(default=""),
    length: str = Form(default="medium"),
    file: UploadFile = File(default=None),
) -> HTMLResponse:
    """Non-streaming fallback used when JS is unavailable."""
    ui_lang = safe_ui_lang(ui_lang)
    style = _normalize_choice(style, VALID_STYLES, "both")
    length = _normalize_choice(length, VALID_LENGTHS, "medium")

    settings_err = _validate_ollama_settings(ui_lang)
    if settings_err:
        return render_result(request, ui_lang, error=settings_err)

    text = (input_text or "").strip()
    upload_path: Optional[str] = None
    if file and file.filename:
        upload_path, ext, err_key = await _save_upload(file)
        if err_key:
            return render_result(request, ui_lang, error=t(ui_lang, err_key))
        if upload_path and ext:
            text = _extract_text(upload_path, ext) or text
            _safe_remove(upload_path)

    if not text:
        return render_result(request, ui_lang, error=t(ui_lang, "no_text"))

    summary, _engine, engine_error = summarize_text(
        text, length=length, style=style, content_lang="auto",
        ollama_url=OLLAMA_URL, model=OLLAMA_MODEL, ollama_api_key=OLLAMA_API_KEY,
    )

    if not summary:
        return render_result(request, ui_lang,
                             error=engine_error or t(ui_lang, "error_ollama"),
                             original=text)

    return render_result(request, ui_lang, engine_error=engine_error,
                         original=text, summary=markdown_to_html(summary))


def _sse(event: dict) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")


@app.post("/summarize-stream")
async def summarize_stream(
    request: Request,
    ui_lang: str = Form(default="en"),
    style: str = Form(default="both"),
    input_text: str = Form(default=""),
    length: str = Form(default="medium"),
    file: UploadFile = File(default=None),
) -> StreamingResponse:
    """Server-Sent Events endpoint emitting status updates while summarizing."""
    ui_lang = safe_ui_lang(ui_lang)
    style = _normalize_choice(style, VALID_STYLES, "both")
    length = _normalize_choice(length, VALID_LENGTHS, "medium")

    # Read upload synchronously here so we can stream events about extraction.
    upload_path: Optional[str] = None
    upload_ext: Optional[str] = None
    upload_err: Optional[str] = None
    if file and file.filename:
        upload_path, upload_ext, upload_err = await _save_upload(file)

    initial_text = (input_text or "").strip()

    async def event_generator():
        path = upload_path
        try:
            settings_err = _validate_ollama_settings(ui_lang)
            if settings_err:
                yield _sse({"step": "error", "message": settings_err})
                return

            yield _sse({"step": "received"})

            text = initial_text
            if upload_err:
                yield _sse({"step": "error", "message": t(ui_lang, upload_err)})
                return
            if path and upload_ext:
                yield _sse({"step": "extracting"})
                # offload blocking extraction
                extracted = await asyncio.to_thread(_extract_text, path, upload_ext)
                _safe_remove(path)
                path = None
                if extracted:
                    text = extracted
                    yield _sse({"step": "extracted", "chars": len(extracted)})
                else:
                    yield _sse({"step": "error", "message": t(ui_lang, "no_text")})
                    return

            if not text:
                yield _sse({"step": "error", "message": t(ui_lang, "no_text")})
                return

            cancelled = {"value": False}

            def is_cancelled() -> bool:
                return cancelled["value"]

            # Run blocking generator in a thread, ferry events through a queue.
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def producer() -> None:
                try:
                    for evt in summarize_text_stream(
                        text, length=length, style=style, content_lang="auto",
                        ollama_url=OLLAMA_URL, model=OLLAMA_MODEL,
                        ollama_api_key=OLLAMA_API_KEY,
                        is_cancelled=is_cancelled,
                    ):
                        if evt.get("step") == "done":
                            evt = {**evt, "original": text}
                        loop.call_soon_threadsafe(queue.put_nowait, evt)
                except Exception as e:
                    logger.exception("Producer crashed")
                    loop.call_soon_threadsafe(queue.put_nowait,
                                              {"step": "error", "message": str(e)})
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            task = asyncio.create_task(asyncio.to_thread(producer))

            try:
                while True:
                    try:
                        evt = await asyncio.wait_for(queue.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            cancelled["value"] = True
                            logger.info("Client disconnected; cancelling summarization")
                            break
                        # heartbeat keeps proxies from closing the stream
                        yield b": ping\n\n"
                        continue

                    if evt is None:
                        break

                    yield _sse(evt)

                    if evt.get("step") in ("done", "error", "cancelled"):
                        break

                    if await request.is_disconnected():
                        cancelled["value"] = True
                        break
            finally:
                cancelled["value"] = True
                try:
                    await task
                except Exception:
                    logger.exception("Producer task error")
        finally:
            _safe_remove(path)

    headers = {
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers=headers)


# ---------------------------------------------------------------------------
# PDF export (Quill HTML -> FPDF compatible)
# ---------------------------------------------------------------------------
def sanitize_text_for_pdf(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "\u202f": " ", "\u00a0": " ", "\u2009": " ", "\u200a": " ",
        "\u200b": "", "\u2028": "\n", "\u2029": "\n", "\ufeff": "",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def pre_process_summary_html(html: str) -> str:
    if not html:
        return ""

    def rgb_to_hex(rgb_str: str) -> Optional[str]:
        match = re.search(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", rgb_str)
        if match:
            return "#{:02x}{:02x}{:02x}".format(int(match.group(1)),
                                                int(match.group(2)),
                                                int(match.group(3)))
        return None

    def replace_align(match: re.Match) -> str:
        tag = match.group(1)
        classes = match.group(2)
        content = match.group(3)
        align = ""
        if "ql-align-center" in classes:
            align = ' align="center"'
        elif "ql-align-right" in classes:
            align = ' align="right"'
        elif "ql-align-justify" in classes:
            align = ' align="justify"'
        return f"<{tag}{align}>{content}</{tag}>"

    html = re.sub(r'<(p|h[1-6]) class="([^"]*ql-align-[^"]*)"[^>]*>(.*?)</\1>',
                  replace_align, html, flags=re.DOTALL)

    def replace_header(match: re.Match) -> str:
        level = match.group(1)
        content = match.group(2)
        size = {"1": "7", "2": "6", "3": "5"}.get(level, "4")
        align_match = re.search(r'align="([^"]*)"', match.group(0))
        align_attr = f' align="{align_match.group(1)}"' if align_match else ""
        return f'<h{level}{align_attr}><font size="{size}"><b>{content}</b></font></h{level}>'

    html = re.sub(r"<h([1-3])(?:[^>]*)>(.*?)</h\1>", replace_header, html, flags=re.DOTALL)

    def replace_span(match: re.Match) -> str:
        attributes = match.group(1)
        content = match.group(2)
        before = ""
        after = ""

        bg_match = re.search(r"background-color:\s*(rgb\(\d+,\s*\d+,\s*\d+\))", attributes)
        if bg_match:
            hex_bg = rgb_to_hex(bg_match.group(1))
            if hex_bg:
                before += f'<mark style="background-color: {hex_bg};">'
                after = "</mark>" + after

        color_match = re.search(r"(?<!background-)color:\s*(rgb\(\d+,\s*\d+,\s*\d+\))", attributes)
        if color_match:
            hex_color = rgb_to_hex(color_match.group(1))
            if hex_color:
                before += f'<font color="{hex_color}">'
                after = "</font>" + after

        if "ql-size-small" in attributes:
            before += '<font size="2">'
            after = "</font>" + after
        elif "ql-size-large" in attributes:
            before += '<font size="5">'
            after = "</font>" + after
        elif "ql-size-huge" in attributes:
            before += '<font size="7">'
            after = "</font>" + after

        return f"{before}{content}{after}"

    html = re.sub(r"<span ([^>]*)>(.*?)</span>", replace_span, html, flags=re.DOTALL)
    return html


@app.post("/export-pdf")
async def export_pdf(
    summary: str = Form(default=""),
    original: str = Form(default=""),
    ui_lang: str = Form(default="en"),
) -> Response:
    pdf = FPDF()
    pdf.add_page()

    font_path = os.path.join(BASE_DIR, "static", "fonts")
    if os.path.exists(os.path.join(font_path, "DejaVuSans.ttf")):
        pdf.add_font("DejaVu", "", os.path.join(font_path, "DejaVuSans.ttf"), uni=True)
        pdf.set_font("DejaVu", size=12)
        use_unicode_font = True
    else:
        pdf.set_font("Helvetica", size=12)
        use_unicode_font = False

    if not use_unicode_font:
        summary = sanitize_text_for_pdf(summary)

    if "<" in summary and ">" in summary:
        try:
            processed = pre_process_summary_html(summary)
            pdf.write_html(
                f'<div style="font-size: 12pt; line-height: 1.5; color: #000000;">{processed}</div>'
            )
        except Exception:
            logger.exception("write_html failed; falling back to multi_cell")
            pdf.multi_cell(0, 6, summary or "")
    else:
        pdf.multi_cell(0, 6, summary or "")

    pdf_bytes = bytes(pdf.output())
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=summary.pdf"},
    )
