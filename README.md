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

4. Each answer is graded by **Grok** (LLM-as-judge) on a **0 or 1 scale**:
   - `1` — correct (semantically equivalent to gold answer)
   - `0` — wrong or empty

## Environment variables (Secrets)

| Variable | Description |
|---|---|
| `XAI_API_KEY` | Your xAI API key (required for judging) |
| `HF_TOKEN` | HuggingFace token (for gold answers dataset + leaderboard upload) |
| `GOLD_DATASET_ID` | HF dataset with gold answers (default: `datakomarov/RAG-data-v2`) |
| `GOLD_FILENAME` | Filename in the dataset (default: `answers_gold.jsonl`) |
| `THIS_SPACE_ID` | This Space's repo ID, e.g. `datakomarov/RAG-LB-v2` |
| `EVAL_MODEL` | Grok model to use (default: `grok-4-1-fast-reasoning`) |
| `EVAL_CONCURRENCY` | Parallel judge calls (default: `5`) |

## Gold answer format

Store your gold answers in a **private** HF dataset:

```json
{"id": "19-1", "question": "Какую модель использовал Николай Кобало?", "answer": "Модель SEIR...", "context": "Опциональный контекст из корпуса..."}
{"id": "14-3", "question": "Как тимлид может поддерживать мотивацию?", "answer": "Декомпозировать задачи..."}
```
Поля `question` и `context` опциональны, но рекомендуются — судья использует их при оценке.
