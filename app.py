import os, uuid, re
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fpdf import FPDF

class MyFPDF(FPDF):
    pass

from file_extractors import extract_text_from_pdf, extract_text_from_docx
from summarizer import summarize_text
from i18n import t
import requests

# Load settings from .env
SUMMARIZATION_ENGINE = os.getenv("SUMMARIZATION_ENGINE", "ollama_local")
CONTENT_LANGUAGE = os.getenv("CONTENT_LANGUAGE", "auto")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Text Summarization System")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

def word_count(s: str) -> int:
    return len((s or "").split())

def get_ratio(length: str) -> float:
    return {"short":0.15, "medium":0.25, "long":0.40}.get(length, 0.25)

def safe_ui_lang(lang: str) -> str:
    return lang if lang in ("en","ar") else "en"

@app.get("/", response_class=HTMLResponse)
def index(request: Request, ui_lang: str = "en"):
    ui_lang = safe_ui_lang(ui_lang)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "ui_lang": ui_lang,
            "t": lambda k: t(ui_lang, k),
        }
    )
@app.get("/api/models")
def get_cloud_models():
    try:
        resp = requests.get("https://ollama.com/api/tags", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"models": [], "error": str(e)}

@app.post("/summarize", response_class=HTMLResponse)
async def summarize(
    request: Request,
    ui_lang: str = Form(default="en"),
    mode: str = Form(default="fallback"),
    style: str = Form(default="both"),
    input_text: str = Form(default=""),
    length: str = Form(default="medium"),
    file: UploadFile = File(default=None),
):
    # Use settings from .env
    engine = SUMMARIZATION_ENGINE
    content_lang = CONTENT_LANGUAGE
    ollama_url = OLLAMA_URL
    model = OLLAMA_MODEL
    ollama_api_key = OLLAMA_API_KEY
    
    ui_lang = safe_ui_lang(ui_lang)
    content_lang = content_lang if content_lang in ("auto","en","ar") else "auto"
    engine = engine if engine in ("ollama_local","ollama_cloud","ollama","extractive") else "ollama_local"
    if engine == "ollama":
        engine = "ollama_local"
    mode = mode if mode in ("strict","fallback") else "fallback"
    style = style if style in ("paragraph","bullets","both") else "both"

    text = (input_text or "").strip()
    
    # Server-side validation for Cloud mode
    if engine == "ollama_cloud":
        if not (ollama_url or "").strip() or not (model or "").strip():
            return templates.TemplateResponse("result.html", {
                "request":request,"ui_lang":ui_lang,"t":lambda k: t(ui_lang,k),
                "error": t(ui_lang,"err_cloud_requires_url_model"),"engine_error":"",
                "original":text,"summary":"",
                "wc_original":word_count(text),"wc_summary":0,"detected_lang":"","content_lang":content_lang,
                "length":length,"engine":engine,"used_engine":engine,"mode":mode,"style":style,
                "model":model
            })
        if not (ollama_api_key or "").strip():
            return templates.TemplateResponse("result.html", {
                "request":request,"ui_lang":ui_lang,"t":lambda k: t(ui_lang,k),
                "error": t(ui_lang, "err_cloud_requires_key"),"engine_error":"",
                "original":text,"summary":"",
                "wc_original":word_count(text),"wc_summary":0,"detected_lang":"","content_lang":content_lang,
                "length":length,"engine":engine,"used_engine":engine,"mode":mode,"style":style,
                "model":model
            })

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
            "model":model
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
            "model":model
        })
    
    # General check: if summary is empty, show error instead of empty result
    if not summary:
        error_msg = engine_error if engine_error else t(ui_lang,"no_text")
        return templates.TemplateResponse("result.html", {
            "request":request,"ui_lang":ui_lang,"t":lambda k: t(ui_lang,k),
            "error": error_msg,"engine_error":"","original":text,"summary":"",
            "wc_original":word_count(text),"wc_summary":0,"detected_lang":detected_lang,"content_lang":content_lang,
            "length":length,"engine":engine,"used_engine":used_engine,"mode":mode,"style":style,
            "model":model
        })

    return templates.TemplateResponse("result.html", {
        "request":request,"ui_lang":ui_lang,"t":lambda k: t(ui_lang,k),
        "error":"", "engine_error": engine_error, "original":text, "summary":summary,
        "wc_original":word_count(text),"wc_summary":word_count(summary),
        "detected_lang":detected_lang,"content_lang":content_lang,
        "length":length,"engine":engine,"used_engine":used_engine,"mode":mode,"style":style,
        "model":model
    })

def sanitize_text_for_pdf(text: str) -> str:
    """Sanitize text for PDF export by replacing problematic Unicode characters."""
    if not text:
        return ""
    # Replace narrow no-break space and other problematic Unicode spaces
    replacements = {
        '\u202f': ' ',  # Narrow no-break space
        '\u00a0': ' ',  # No-break space
        '\u2009': ' ',  # Thin space
        '\u200a': ' ',  # Hair space
        '\u200b': '',   # Zero-width space
        '\u2028': '\n', # Line separator
        '\u2029': '\n', # Paragraph separator
        '\ufeff': '',   # BOM
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Remove any remaining characters that can't be encoded in latin-1
    return text.encode('latin-1', errors='replace').decode('latin-1')

def pre_process_summary_html(html: str) -> str:
    """
    Convert Quill-specific HTML styles to FPDF-compatible tags.
    Handles color, background-color, alignment, font size, strikethrough, and headers.
    """
    if not html:
        return ""

    def rgb_to_hex(rgb_str):
        match = re.search(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', rgb_str)
        if match:
            return '#{:02x}{:02x}{:02x}'.format(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        return None

    # 1. Handle alignment classes in <p> or <hX> tags
    def replace_align(match):
        tag = match.group(1)
        classes = match.group(2)
        content = match.group(3)
        align = ""
        if 'ql-align-center' in classes: align = ' align="center"'
        elif 'ql-align-right' in classes: align = ' align="right"'
        elif 'ql-align-justify' in classes: align = ' align="justify"'
        return f'<{tag}{align}>{content}</{tag}>'

    html = re.sub(r'<(p|h[1-6]) class="([^"]*ql-align-[^"]*)"[^>]*>(.*?)</\1>', replace_align, html, flags=re.DOTALL)

    # 2. Enhance header tags with appropriate font sizes
    # H1 -> size 7, H2 -> size 6, H3 -> size 5
    def replace_header(match):
        level = match.group(1)
        content = match.group(2)
        size_map = {'1': '7', '2': '6', '3': '5'}
        size = size_map.get(level, '4')
        # Preserve any align attribute if present
        align_match = re.search(r'align="([^"]*)"', match.group(0))
        align_attr = f' align="{align_match.group(1)}"' if align_match else ''
        return f'<h{level}{align_attr}><font size="{size}"><b>{content}</b></font></h{level}>'
    
    html = re.sub(r'<h([1-3])(?:[^>]*)>(.*?)</h\1>', replace_header, html, flags=re.DOTALL)

    # 3. Handle spans with styles and/or classes (color, background-color, font-size)
    def replace_span(match):
        attributes = match.group(1)
        content = match.group(2)
        
        tags_before = ""
        tags_after = ""
        
        # Handle background color first (before text color to avoid conflicts)
        bg_match = re.search(r'background-color:\s*(rgb\(\d+,\s*\d+,\s*\d+\))', attributes)
        if bg_match:
            hex_bg = rgb_to_hex(bg_match.group(1))
            if hex_bg:
                # fpdf2 doesn't support bgcolor in font tag well, but we can wrap in a span-like approach
                # For better compatibility, we'll use a marker that can be styled
                tags_before += f'<mark style="background-color: {hex_bg};">'
                tags_after = '</mark>' + tags_after
        
        # Handle text color in style attribute (use negative lookbehind to avoid matching background-color)
        color_match = re.search(r'(?<!background-)color:\s*(rgb\(\d+,\s*\d+,\s*\d+\))', attributes)
        if color_match:
            hex_color = rgb_to_hex(color_match.group(1))
            if hex_color:
                tags_before += f'<font color="{hex_color}">'
                tags_after = '</font>' + tags_after

        # Handle Quill classes for font size
        if 'ql-size-small' in attributes:
            tags_before += '<font size="2">'
            tags_after = '</font>' + tags_after
        elif 'ql-size-large' in attributes:
            tags_before += '<font size="5">'
            tags_after = '</font>' + tags_after
        elif 'ql-size-huge' in attributes:
            tags_before += '<font size="7">'
            tags_after = '</font>' + tags_after

        return f"{tags_before}{content}{tags_after}"

    html = re.sub(r'<span ([^>]*)>(.*?)</span>', replace_span, html, flags=re.DOTALL)
    
    # 4. Ensure strikethrough (<s> or <del>) tags are preserved
    # fpdf2 supports <s> tag natively, so no conversion needed
    # Just ensure they're present in the output
    
    return html

@app.post("/export-pdf")
async def export_pdf(
    summary: str = Form(default=""),
    original: str = Form(default=""),
    ui_lang: str = Form(default="en"),
):
    """Export summary to PDF file"""
    pdf = MyFPDF()
    pdf.add_page()
    
    # Add Unicode font for Arabic support
    font_path = os.path.join(BASE_DIR, "static", "fonts")
    if os.path.exists(os.path.join(font_path, "DejaVuSans.ttf")):
        pdf.add_font("DejaVu", "", os.path.join(font_path, "DejaVuSans.ttf"), uni=True)
        pdf.set_font("DejaVu", size=12)
        use_unicode_font = True
    else:
        pdf.set_font("Helvetica", size=12)
        use_unicode_font = False
    
    # Sanitize text if not using Unicode font
    if not use_unicode_font:
        summary = sanitize_text_for_pdf(summary)
    
    # If summary contains HTML (from Quill), use write_html
    if "<" in summary and ">" in summary:
        try:
            # Pre-process HTML to handle Quill-specific styling (like colors)
            processed_summary = pre_process_summary_html(summary)
            
            # HTMLMixin/fpdf2 has limited CSS support. Use simple tags for better compatibility.
            # Explicitly setting text color to black for lists and base text to avoid 'red' numbers.
            html_wrapper = f"""
            <div style="font-size: 12pt; line-height: 1.5; color: #000000;">
                {processed_summary}
            </div>
            """
            pdf.write_html(html_wrapper)
        except Exception:
            pdf.multi_cell(0, 6, summary or "")
    else:
        pdf.multi_cell(0, 6, summary or "")
    
    # Generate PDF bytes
    pdf_bytes = bytes(pdf.output())
    
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=summary.pdf"}
    )