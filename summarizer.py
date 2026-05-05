"""Ollama-cloud summarization with streaming progress events."""
from __future__ import annotations

import logging
import os
import re
from typing import Callable, Iterator

import ollama

logger = logging.getLogger(__name__)

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT_SECONDS = 120
MAX_CHUNK_CHARS = 8000

NUM_PREDICT_BY_LENGTH = {"short": 220, "medium": 380, "long": 650}


def detect_lang(text: str) -> str:
    if not text:
        return "en"
    arabic_chars = len(ARABIC_RANGE.findall(text))
    return "ar" if arabic_chars >= max(20, int(len(text) * 0.08)) else "en"


def _build_prompt(lang: str, style: str, length: str, chunk_text: str) -> str:
    if lang == "ar":
        style_txt = {
            "paragraph": "اكتب فقرة واحدة فقط.",
            "bullets": "اكتب 5-7 نقاط مختصرة فقط.",
            "both": "اكتب فقرة قصيرة ثم 5-7 نقاط.",
        }[style]
        len_txt = {"short": "قصير", "medium": "متوسط", "long": "طويل"}[length]
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
    style_txt = {
        "paragraph": "Write one paragraph only.",
        "bullets": "Write 5-7 concise bullet points only.",
        "both": "Write a short paragraph then 5-7 bullet points.",
    }[style]
    len_txt = {"short": "short", "medium": "medium", "long": "long"}[length]
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


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks: list[str] = []
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


def _ollama_headers(api_key: str) -> dict:
    api_key = (api_key or "").strip()
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if not api_key:
        return headers
    if api_key.lower().startswith("bearer "):
        headers["Authorization"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _build_client(ollama_url: str, api_key: str) -> ollama.Client:
    return ollama.Client(
        host=ollama_url.rstrip("/"),
        headers=_ollama_headers(api_key),
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )


def ollama_generate(prompt: str, ollama_url: str, model: str, num_predict: int,
                    api_key: str = "", temperature: float = 0.0) -> str:
    client = _build_client(ollama_url, api_key)
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
        options={"temperature": temperature, "num_predict": num_predict},
    )
    return response["message"]["content"].strip()


def _resolve_lang(text: str, content_lang: str) -> str:
    lang = detect_lang(text) if content_lang == "auto" else content_lang
    return lang if lang in ("en", "ar") else "en"


def ollama_summarize(text: str, length: str = "medium", content_lang: str = "auto",
                     style: str = "both", ollama_url: str = DEFAULT_OLLAMA_URL,
                     model: str = DEFAULT_OLLAMA_MODEL, api_key: str = "") -> str:
    text = (text or "").strip()
    if not text:
        return ""
    lang = _resolve_lang(text, content_lang)
    num_predict = NUM_PREDICT_BY_LENGTH.get(length, NUM_PREDICT_BY_LENGTH["medium"])

    chunks = _chunk_text(text)
    partial = [
        ollama_generate(_build_prompt(lang, style, length, ch), ollama_url, model,
                        num_predict=num_predict, api_key=api_key)
        for ch in chunks
    ]
    if len(partial) == 1:
        return partial[0]
    merged = "\n\n".join(partial)
    return ollama_generate(_build_prompt(lang, style, length, merged), ollama_url, model,
                           num_predict=num_predict, api_key=api_key)


def summarize_text(text: str, length: str = "medium", style: str = "both",
                   content_lang: str = "auto",
                   ollama_url: str = DEFAULT_OLLAMA_URL,
                   model: str = DEFAULT_OLLAMA_MODEL,
                   ollama_api_key: str = "") -> tuple[str, str, str]:
    """Non-streaming entry point. Returns (summary, used_engine, error)."""
    text = (text or "").strip()
    if not text:
        return "", "ollama_cloud", ""
    try:
        out = ollama_summarize(
            text, length=length, content_lang=content_lang, style=style,
            ollama_url=ollama_url, model=model, api_key=ollama_api_key,
        )
        return out, "ollama_cloud", ""
    except Exception as e:
        logger.exception("Ollama summarization failed")
        return "", "ollama_cloud", str(e)


def summarize_text_stream(
    text: str,
    length: str = "medium",
    style: str = "both",
    content_lang: str = "auto",
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_OLLAMA_MODEL,
    ollama_api_key: str = "",
    is_cancelled: Callable[[], bool] = lambda: False,
) -> Iterator[dict]:
    """Yield progress events while summarizing.

    Events:
        {"step": "summarizing", "chunk": int, "total": int}
        {"step": "merging"}
        {"step": "done", "summary": str}
        {"step": "error", "message": str}
        {"step": "cancelled"}
    """
    text = (text or "").strip()
    if not text:
        yield {"step": "error", "message": "empty_text"}
        return

    try:
        lang = _resolve_lang(text, content_lang)
        num_predict = NUM_PREDICT_BY_LENGTH.get(length, NUM_PREDICT_BY_LENGTH["medium"])
        chunks = _chunk_text(text)
        total = len(chunks)
        partial: list[str] = []

        for i, ch in enumerate(chunks, start=1):
            if is_cancelled():
                yield {"step": "cancelled"}
                return
            yield {"step": "summarizing", "chunk": i, "total": total}
            prompt = _build_prompt(lang, style, length, ch)
            out = ollama_generate(prompt, ollama_url, model,
                                  num_predict=num_predict, api_key=ollama_api_key)
            partial.append(out)

        if is_cancelled():
            yield {"step": "cancelled"}
            return

        if len(partial) == 1:
            yield {"step": "done", "summary": partial[0]}
            return

        yield {"step": "merging"}
        merged = "\n\n".join(partial)
        final = ollama_generate(_build_prompt(lang, style, length, merged),
                                ollama_url, model, num_predict=num_predict,
                                api_key=ollama_api_key)
        yield {"step": "done", "summary": final}
    except Exception as e:
        logger.exception("Streaming summarization failed")
        yield {"step": "error", "message": str(e)}
