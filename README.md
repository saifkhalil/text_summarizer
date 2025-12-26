# Text Summarizer v6 (Local Ollama)

This version uses **Ollama locally** instead of Hugging Face models.

## 1) Pull a good model (Arabic/English)
Recommended:
```bash
ollama pull qwen2.5:7b-instruct
```
Faster option:
```bash
ollama pull qwen2.5:3b-instruct
```

## 2) Run the web app
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open:
- http://127.0.0.1:8000/?ui_lang=ar

## Environment variables (optional)
- OLLAMA_URL=http://localhost:11434
- OLLAMA_MODEL=qwen2.5:7b-instruct

## If you want to verify Ollama works
```bash
curl http://localhost:11434/api/tags
```
