import os, re, nltk, requests, ollama
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")

def ensure_nltk_resources():
    for res in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
        try:
            nltk.data.find(res)
        except LookupError:
            nltk.download(res.split("/")[-1], quiet=True)

def detect_lang(text: str) -> str:
    if not text:
        return "en"
    arabic_chars = len(ARABIC_RANGE.findall(text))
    return "ar" if arabic_chars >= max(20, int(len(text) * 0.08)) else "en"

def split_sentences_fallback(text: str, lang: str):
    if lang == "ar":
        parts = re.split(r"(?<=[\.\!\؟\!])\s+|\n+", text)
    else:
        parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]

def extractive_summarize(text: str, ratio: float = 0.25):
    text = (text or "").strip()
    if not text:
        return ""
    lang = detect_lang(text)
    sents = split_sentences_fallback(text, lang)
    if len(sents) <= 3:
        return text
    try:
        ensure_nltk_resources()
        tokenizer_lang = "arabic" if lang == "ar" else "english"
        parser = PlaintextParser.from_string(text, Tokenizer(tokenizer_lang))
        target = max(3, int(len(list(parser.document.sentences)) * ratio))
        target = min(target, 12)
        summarizer = LsaSummarizer()
        summary_sents = summarizer(parser.document, target)
        out = " ".join(str(s) for s in summary_sents).strip()
        if out:
            return out
    except Exception:
        pass
    target = max(3, int(len(sents) * ratio))
    target = min(target, 12, len(sents))
    return " ".join(sents[:target]).strip()

def _build_prompt(lang: str, style: str, length: str, chunk_text: str):
    if lang == "ar":
        style_txt = {
            "paragraph": "اكتب فقرة واحدة فقط.",
            "bullets": "اكتب 5-7 نقاط مختصرة فقط.",
            "both": "اكتب فقرة قصيرة ثم 5-7 نقاط.",
        }[style]
        len_txt = {"short":"قصير", "medium":"متوسط", "long":"طويل"}[length]
        return (
            "أنت مساعد تلخيص محترف.\n"
            "المطلوب: لخص النص التالي باللغة العربية الفصحى.\n"
            f"الطول: {len_txt}. {style_txt}\n"
            "قيود مهمة:\n"
            "- لا تضف أي معلومات غير موجودة بالنص.\n"
            "- لا تخترع أسماء أو تواريخ أو أحداث.\n"
            "- حافظ على الأرقام والتواريخ كما هي إن وُجدت.\n\n"
            "النص:\n"
            f"{chunk_text}\n\n"
            "الملخص:"
        )
    else:
        style_txt = {
            "paragraph":"Write one paragraph only.",
            "bullets":"Write 5-7 concise bullet points only.",
            "both":"Write a short paragraph then 5-7 bullet points.",
        }[style]
        len_txt = {"short":"short", "medium":"medium", "long":"long"}[length]
        return (
            "You are a professional summarization assistant.\n"
            "Task: Summarize the following text.\n"
            f"Length: {len_txt}. {style_txt}\n"
            "Important constraints:\n"
            "- Do NOT add facts not present in the text.\n"
            "- Do NOT invent names, dates, or events.\n"
            "- Preserve numbers/dates if present.\n\n"
            "Text:\n"
            f"{chunk_text}\n\n"
            "Summary:"
        )

def _chunk_text(text: str, max_chars: int = 8000):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end]
        cut = chunk.rfind("\n")
        if cut > max_chars * 0.6:
            chunk = chunk[:cut]
            end = start + cut
        chunks.append(chunk.strip())
        start = end
    return [c for c in chunks if c]

def ollama_generate(prompt: str, ollama_url: str, model: str, num_predict: int, api_key: str = "", temperature: float = 0.0):
    url = ollama_url.rstrip("/")
    is_cloud = "ollama.com" in url.lower()
    
    headers = _ollama_headers(api_key)
    client = ollama.Client(host=url, headers=headers)
    
    options = {
        "temperature": temperature,
        "num_predict": num_predict,
    }

    if is_cloud:
        # Using chat API for cloud as recommended
        response = client.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False,
            options=options
        )
        return response['message']['content'].strip()
    else:
        # Standard generate for local/other
        response = client.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options=options
        )
        return response['response'].strip()

def ollama_summarize(text: str, length: str = "medium", content_lang: str = "auto",
                    style: str = "both", ollama_url: str = DEFAULT_OLLAMA_URL,
                    model: str = DEFAULT_OLLAMA_MODEL, api_key: str = ""):
    text = (text or "").strip()
    if not text:
        return "", "en"
    lang = detect_lang(text) if content_lang == "auto" else content_lang

    num_predict = {"short":220, "medium":380, "long":650}.get(length, 380)

    chunks = _chunk_text(text, max_chars=8000)
    partial = []
    for ch in chunks:
        prompt = _build_prompt(lang, style, length, ch)
        partial.append(ollama_generate(prompt, ollama_url, model, num_predict=num_predict, api_key=api_key))

    if len(partial) == 1:
        return partial[0], lang

    merged = "\n\n".join(partial)
    reduce_prompt = _build_prompt(lang, style, length, merged)
    final = ollama_generate(reduce_prompt, ollama_url, model, num_predict=num_predict, api_key=api_key)
    return final, lang

def summarize_text(text: str, ratio: float = 0.25, length: str = "medium", content_lang: str = "auto",
                   engine: str = "ollama_local", mode: str = "fallback", style: str = "both",
                   ollama_url: str = DEFAULT_OLLAMA_URL, model: str = DEFAULT_OLLAMA_MODEL,
                   ollama_api_key: str = ""):
    text = (text or "").strip()
    if not text:
        return "", "en", engine, ""

    detected = detect_lang(text) if content_lang == "auto" else content_lang

    if engine == "extractive":
        return extractive_summarize(text, ratio=ratio), detected, "extractive", ""

    try:
        out, _lang = ollama_summarize(
            text, length=length, content_lang=content_lang, style=style,
            ollama_url=ollama_url, model=model, api_key=ollama_api_key
        )
        return out, detected, "ollama", ""
    except Exception as e:
        if mode == "strict":
            return "", detected, "ollama", str(e)
        return extractive_summarize(text, ratio=ratio), detected, "extractive", str(e)


def _ollama_headers(api_key: str) -> dict:
    api_key = (api_key or "").strip()
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    if not api_key:
        return headers
    
    if api_key.lower().startswith("bearer "):
        headers["Authorization"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
