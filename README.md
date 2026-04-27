---
title: RAG Leaderboard v2.1
emoji: 🏁
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# RAG Leaderboard v2

Leaderboard for evaluating RAG (Retrieval-Augmented Generation) systems.

## How it works

1. Download the public question set from `data/questions/questions_public.jsonl`
2. Run your RAG pipeline and generate answers
3. Upload a JSONL file with your answers — one JSON object per line:

```json
{"id": "0", "answer": "Your answer here"}
{"id": "1", "answer": "Another answer"}
```

4. Each answer is graded by an **LLM-as-judge** on a **0 or 1 scale**:
   - `1` — correct (semantically equivalent to gold answer)
   - `0` — wrong or empty

## Local deployment with Docker

```bash
# Create .env file with your variables
cp .env.example .env
nano .env

# Create data directories
mkdir -p data/questions data/results data/documents

# Start with Docker Compose
docker compose up -d

# View logs
docker compose logs -f
```

The app will be available at http://localhost:7860

```bash
# Create .env file with your variables
cp .env.example .env
nano .env

# Create data directories
mkdir -p data/questions data/results data/documents

# Start with Docker Compose
docker compose up -d

# View logs
docker compose logs -f
```

### Data persistence

The following directories are mounted as volumes:

| Host path | Container path | Content |
|---|---|---|
| `./data/questions` | `/app/data/questions` | Question sets |
| `./data/results` | `/app/data/results` | `leaderboard.csv`, `eval_details.jsonl` |
| `./data/documents` | `/app/data/documents` | Source documents |

## Environment variables (Secrets)

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI-compatible API key (required for judging) |
| `HF_TOKEN` | HuggingFace token (for gold answers dataset + leaderboard upload) |
| `GOLD_DATASET_ID` | HF dataset with gold answers (default: `datakomarov/RAG-data-v2`) |
| `GOLD_FILENAME` | Filename in the dataset (default: `answers_gold.jsonl`) |
| `THIS_SPACE_ID` | This Space's repo ID, e.g. `datakomarov/RAG-LB-v2` |
| `EVAL_MODEL` | Judge model to use (default: `grok-4-1-fast-reasoning`) |
| `EVAL_CONCURRENCY` | Parallel judge calls (default: `5`) |
| `OPENAI_BASE_URL` | Base URL for OpenAI-compatible API (default: `https://api.x.ai/v1`) |
| `SHOW_DETAILS` | Show evaluation details in UI (default: `0`) |

## Gold answer format

Store your gold answers in a **private** HF dataset:

```json
{"id": "19-1", "question": "Какую модель использовал Николай Кобало?", "answer": "Модель SEIR...", "context": "Опциональный контекст из корпуса..."}
{"id": "14-3", "question": "Как тимлид может поддерживать мотивацию?", "answer": "Декомпозировать задачи..."}
```
Поля `question` и `context` опциональны, но рекомендуются — судья использует их при оценке.
