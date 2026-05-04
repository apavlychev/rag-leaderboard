#!/usr/bin/env python3
"""
Читает текстовые файлы из указанной директории,
для каждого генерирует вопросы, ответы и контекст через OpenAI-compatible API
с использованием Structured Outputs (response_format с JSON Schema),
и сохраняет результат в .jsonl файл с тем же именем.

Большие файлы (> CHUNK_THRESHOLD символов) автоматически бьются на чанки.

Использование:
    python md_to_qa.py
    python md_to_qa.py --model gpt-4o --extensions md,txt --merge
    python md_to_qa.py --directory ./data --extensions txt

Требования:
    pip install openai python-dotenv

Переменные окружения:
    OPENAI_API_KEY — ваш API ключ (обязательно)
    EVAL_MODEL     — модель по умолчанию (опционально)
    OPENAI_BASE_URL — базовый URL API (опционально)
"""

import argparse
import hashlib
import logging
import os
import json
import glob
import time

from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from dotenv import load_dotenv

# Загружаем .env (по умолчанию ищет файл .env)
load_dotenv()


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Настройки
# ──────────────────────────────────────────────
CHARS_PER_QUESTION  = 1000  # ~1 вопрос на 1000 символов
MIN_QUESTIONS_TOTAL = 1     # минимум вопросов на документ
MAX_QUESTIONS_CHUNK = 50    # максимум вопросов за один запрос (один чанк)
CONTEXT_MAX_TOKENS  = 512
REQUEST_TIMEOUT    = 600

MAX_RETRIES        = 3
RETRY_BASE_DELAY   = 5
RATE_LIMIT_DELAY   = 15

CHUNK_THRESHOLD    = 50_000
CHUNK_SIZE         = 40_000

# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Генерирует Q&A пары из текстовых файлов через OpenAI-compatible API"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Модель для генерации (переопределяет EVAL_MODEL из env)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Объединить все результаты в один файл result.jsonl"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default="md",
        help="Расширения файлов через запятую (например: md,txt,rst)"
    )
    parser.add_argument(
        "--directory",
        type=str,
        default="./data/utils",
        help="Директория с файлами для обработки"
    )
    return parser.parse_args()


def get_model(cli_model: str | None) -> str:
    """Возвращает модель: из CLI > env > default."""
    if cli_model:
        return cli_model
    return os.getenv("EVAL_MODEL", "grok-4-1-fast-reasoning")

# ──────────────────────────────────────────────
# JSON Schema для structured output
# ──────────────────────────────────────────────
QA_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "qa_list",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "Вопрос для RAG-оценки"
                            },
                            "answer": {
                                "type": "string",
                                "description": "Развёрнутый ответ на основе текста"
                            },
                            "context": {
                                "type": "string",
                                "description": "Дословная цитата из текста с ответом"
                            }
                        },
                        "required": ["question", "answer", "context"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["items"],
            "additionalProperties": False
        }
    }
}


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "Переменная окружения OPENAI_API_KEY не задана.\n"
            "Задайте её: export OPENAI_API_KEY='your_key_here'"
        )
    return key


def get_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", "https://api.x.ai/v1")


def find_text_files(directory: str, extensions: list[str]) -> list[str]:
    """Находит все файлы с указанными расширениями."""
    files = []
    for ext in extensions:
        pattern = os.path.join(directory, f"*.{ext.lstrip('.')}")
        files.extend(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"Нет файлов с расширениями {extensions} в директории: {directory}")

    return sorted(set(files))  # уникальные, отсортированные


def merge_results(output_files: list[str], directory: str) -> None:
    """Объединяет все jsonl файлы в один в указанной директории."""
    merged = []
    for path in output_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    merged.append(json.loads(line))

    merge_path = os.path.join(directory, "result.jsonl")
    with open(merge_path, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("📦 Объединённый файл: %s (%d записей)", merge_path, len(merged))


def split_into_chunks(text: str, chunk_size: int) -> list[str]:
    """Бьёт текст на куски по chunk_size символов, стараясь резать по абзацам."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        cut = text.rfind("\n\n", start, end)
        if cut <= start:
            cut = text.rfind("\n", start, end)
        if cut <= start:
            cut = end
        chunks.append(text[start:cut])
        start = cut
    return [c.strip() for c in chunks if c.strip()]


def simple_hash(text: str) -> str:
    """Простой детерминированный хеш для генерации ID."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def call_with_retry(client: OpenAI, **kwargs) -> object:
    """Запрос к API с повторами при временных ошибках."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)

        except RateLimitError:
            if attempt == MAX_RETRIES:
                raise
            logger.warning("Rate limit (попытка %d/%d), ждём %d сек...", attempt, MAX_RETRIES, RATE_LIMIT_DELAY)
            time.sleep(RATE_LIMIT_DELAY)

        except (APIConnectionError, APITimeoutError, InternalServerError) as e:
            if attempt == MAX_RETRIES:
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.warning("%s (попытка %d/%d), ждём %d сек...", type(e).__name__, attempt, MAX_RETRIES, delay)
            time.sleep(delay)


def get_doc_identity(client: OpenAI, text: str, model: str) -> str:
    """Шаг 1: кратко описываем документ для контекста при формулировке вопросов."""
    prompt = f"""Прочитай текст ниже и ответь одним абзацем (3-5 предложений):
- Кто автор (если указан)?
- Какова тема работы?
- Какой тип документа (курсовая, магистерская диссертация, научная статья, учебная программа и т.п.)?
- Какие ключевые методы, модели или результаты упоминаются?

Ответ нужен для формулировки вопросов вида
"В работе [автор] по теме [тема] — какой метод был использован для X?"

ТЕКСТ (первые 3000 символов):
{text[:3000]}
"""
    response = call_with_retry(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=120,
    )
    return response.choices[0].message.content.strip()


def generate_qa_for_chunk(client: OpenAI, chunk: str, doc_identity: str, num_questions: int, model: str) -> list[dict]:
    """Генерирует Q&A для одного куска текста через structured output."""
    prompt = f"""Ты — эксперт по созданию датасетов для оценки RAG-систем.

ОПИСАНИЕ ДОКУМЕНТА (используй для формулировки вопросов):
{doc_identity}

ЗАДАЧА: по тексту ниже сгенерируй ровно {num_questions} вопросов с ответами.

ТРЕБОВАНИЯ К ВОПРОСАМ:
1. Вопрос должен идентифицировать работу через автора, тему или контекст —
   без указания на документ или его структуру.
   Плохо: "Какой ROC AUC у LightGBM?"  ← непонятно в какой работе
   Плохо: "Что написано в разделе 3?"  ← ссылка на структуру
   Хорошо: "Какой ROC AUC показал LightGBM в работе Фролова по кредитному скорингу Сбербанка?"
   Хорошо: "Какую архитектуру нейросети использовал Фролов для обработки транзакций заемщиков?"

2. Вопросы должны охватывать разные аспекты: методологию, данные, результаты, выводы.

3. Избегай тривиальных вопросов "что такое X?" для общеизвестных терминов.

Поле "context" — дословная цитата из текста *БЕЗ СОКРАЩЕНИЙ ТИПА ...*, где содержится ответ (максимум {CONTEXT_MAX_TOKENS} токенов).

ТЕКСТ:
{chunk}
"""

    response = call_with_retry(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format=QA_SCHEMA,
        timeout=REQUEST_TIMEOUT,
    )

    data = json.loads(response.choices[0].message.content)
    return data["items"]


def process_file(client: OpenAI, md_path: str, model: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    char_count = len(text)
    total_questions = max(MIN_QUESTIONS_TOTAL, char_count // CHARS_PER_QUESTION)
    logger.info("   Символов: %d → вопросов: %d", char_count, total_questions)

    # Шаг 1: описание документа
    logger.info("   Шаг 1: определяем тип и контекст документа...")
    doc_identity = get_doc_identity(client, text, model)
    logger.info("   ✓ → %s...", doc_identity[:120].replace(chr(10), " "))

    # Шаг 2: генерация Q&A
    use_chunks = char_count > CHUNK_THRESHOLD
    all_qa = []

    if not use_chunks:
        n_q = min(total_questions, MAX_QUESTIONS_CHUNK)
        logger.info("   Шаг 2: генерируем %d вопросов...", n_q)
        t0 = time.time()
        all_qa = generate_qa_for_chunk(client, text, doc_identity, n_q, model)
        logger.info("   ✓ получено пар: %d (%.1f сек)", len(all_qa), time.time() - t0)
    else:
        chunks = split_into_chunks(text, CHUNK_SIZE)
        logger.info("   Шаг 2: файл большой, разбит на %d чанков", len(chunks))

        total_chars = sum(len(c) for c in chunks)
        distributed = 0
        for i, chunk in enumerate(chunks, 1):
            if i == len(chunks):
                n_q = total_questions - distributed
            else:
                n_q = max(1, round(total_questions * len(chunk) / total_chars))
            n_q = min(n_q, MAX_QUESTIONS_CHUNK)  # не больше лимита за один запрос
            distributed += n_q

            logger.info("   Чанк %d/%d (%d симв.) → %d вопросов...", i, len(chunks), len(chunk), n_q)
            t0 = time.time()
            try:
                qa = generate_qa_for_chunk(client, chunk, doc_identity, n_q, model)
                all_qa.extend(qa)
                logger.info("   ✓ получено %d (%.1f сек)", len(qa), time.time() - t0)
            except Exception:
                logger.exception("   ❌ пропущен")

    logger.info("   Итого пар: %d", len(all_qa))

    filename = os.path.basename(md_path)
    base_name = os.path.splitext(md_path)[0]
    out_path = base_name + ".jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for item in all_qa:
            item["documentId"] = os.path.splitext(filename)[0]
            item["id"] = simple_hash(item["question"])
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return out_path


def main():
    args = parse_args()

    # Настройки из аргументов
    model = get_model(args.model)
    extensions = [e.strip() for e in args.extensions.split(",")]
    directory = os.path.abspath(args.directory)

    logger.info("📂 Директория: %s", directory)
    logger.info("🤖 Модель: %s", model)
    logger.info("📋 Расширения: %s", ", ".join(extensions))

    files = find_text_files(directory, extensions)
    logger.info("📄 Найдено файлов: %d", len(files))

    client = OpenAI(
        api_key=get_api_key(),
        base_url=get_base_url(),
    )

    errors = []
    output_files = []  # для merge

    for i, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        logger.info("[%d/%d] 📄 %s", i, len(files), filename)
        try:
            out_path = process_file(client, file_path, model)
            output_files.append(out_path)
            logger.info("   💾 Сохранено: %s", os.path.basename(out_path))
        except Exception:
            logger.exception("   ❌ Ошибка")
            errors.append((filename, "см. stacktrace выше"))

    # Merge если указан флаг
    if args.merge and output_files:
        merge_results(output_files, directory)

    logger.info("─" * 40)
    logger.info("✅ Обработано: %d/%d", len(files) - len(errors), len(files))
    if errors:
        logger.error("❌ Ошибки:")
        for fname, err in errors:
            logger.error("   %s: %s", fname, err)


if __name__ == "__main__":
    main()
