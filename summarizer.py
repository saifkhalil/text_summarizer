import re
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    # NLTK new versions need this too
    try:
        nltk.data.find("tokenizers/punkt_tab/english")
    except LookupError:
        nltk.download("punkt_tab")


ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")

def detect_lang(text: str) -> str:
    # heuristic: if a decent portion of chars are Arabic, treat as Arabic
    if not text:
        return "en"
    arabic_chars = len(ARABIC_RANGE.findall(text))
    return "ar" if arabic_chars >= max(20, int(len(text) * 0.08)) else "en"

def normalize(text: str) -> str:
    return (text or "").strip()

def split_sentences_fallback(text: str, lang: str) -> list[str]:
    # fallback sentence splitting
    if lang == "ar":
        # Arabic punctuation: . ؟ !
        parts = re.split(r"(?<=[\.\!\؟\!])\s+|\n+", text)
    else:
        parts = re.split(r"(?<=[\.\!\?])\s+|\n+", text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts

def get_tokenizer_lang(lang: str) -> str:
    # Sumy tokenizers are limited; we use english for non-supported but keep fallback splitting.
    return "english" if lang != "en" else "english"

def summarize_text(text: str, ratio: float = 0.25, content_lang: str = "auto") -> tuple[str, str]:
    text = normalize(text)
    if not text:
        return "", "en"

    lang = detect_lang(text) if content_lang == "auto" else content_lang

    # If very short, return as-is
    sents = split_sentences_fallback(text, lang)
    if len(sents) <= 3:
        return text, lang

    # LSA summarization using Sumy
    tokenizer = Tokenizer(get_tokenizer_lang(lang))
    ensure_nltk()
    parser = PlaintextParser.from_string(text, tokenizer)

    sentences_count = len(list(parser.document.sentences))
    if sentences_count <= 3:
        return text, lang

    target = max(3, int(sentences_count * ratio))
    target = min(target, 12)

    summarizer = LsaSummarizer()
    summary_sents = summarizer(parser.document, target)
    summary = " ".join(str(s) for s in summary_sents).strip()

    # If summary came empty (rare), fallback: first N sentences
    if not summary:
        summary = " ".join(sents[:target])

    return summary, lang
