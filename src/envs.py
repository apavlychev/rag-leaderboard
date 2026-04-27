import os
import json

from dotenv import load_dotenv

# Загружаем .env (по умолчанию ищет файл .env)
load_dotenv()

# ── OpenAI-Compatible API ───────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EVAL_MODEL = os.getenv("EVAL_MODEL", "grok-4-1-fast-reasoning")
EVAL_CONCURRENCY = int(os.getenv("EVAL_CONCURRENCY", "5"))
PROXY_URL = os.getenv("PROXY_URL", "")

# ── OpenAI-compatible API ─────────────────────────────────────────────────────
# Base URL for OpenAI-compatible API
# Default: https://api.x.ai/v1 (xAI Grok, or any OpenAI-compatible endpoint)
# Other examples:
#   - OpenAI: https://api.openai.com/v1
#   - DeepSeek: https://api.deepseek.com/v1
#   - Together AI: https://api.together.xyz/v1
#   - Local LLM: http://localhost:8000/v1
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.x.ai/v1")

# ── Пути к данным ─────────────────────────────────────────────────────────────
QUESTIONS_PATH = os.getenv("QUESTIONS_PATH", "data/questions/questions_public.jsonl")

# Эталонные ответы — параметры загрузки
_HF_TOKEN = os.getenv("HF_TOKEN", "")
_GOLD_DATASET_ID = os.getenv("GOLD_DATASET_ID", "datakomarov/RAG-data-v2")
_GOLD_FILENAME = os.getenv("GOLD_FILENAME", "answers_gold.jsonl")

# Ленивая загрузка — вызывается только при первом evaluate_submission
_gold_path_cache = None

def get_gold_path() -> str:
    """Возвращает путь к gold-файлу, загружая его при первом вызове."""
    global _gold_path_cache
    if _gold_path_cache is not None:
        return _gold_path_cache

    local_override = os.getenv("GOLD_PATH_LOCAL", "")
    if local_override and os.path.exists(local_override):
        _gold_path_cache = local_override
        return _gold_path_cache

    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=_GOLD_DATASET_ID,
            filename=_GOLD_FILENAME,
            repo_type="dataset",
            token=_HF_TOKEN,
            local_dir=".",
        )
        _gold_path_cache = path
        return _gold_path_cache
    except Exception as e:
        raise RuntimeError(
            f"Cannot load gold answers from '{_GOLD_DATASET_ID}/{_GOLD_FILENAME}'. "
            f"Set GOLD_PATH_LOCAL env var to use a local file. Error: {e}"
        )


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
