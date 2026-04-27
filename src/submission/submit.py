# src/submission/submit.py  — LLM-as-judge через OpenAI-compatible API
import json
import logging
import re
import httpx
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.envs import (
    OPENAI_API_KEY,
    EVAL_MODEL,
    EVAL_CONCURRENCY,
    PROXY_URL,
    OPENAI_BASE_URL,
    QUESTIONS_PATH,
    get_gold_path,
    load_jsonl,
)

# ── Logging ─────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Клиент xAI (ленивая инициализация) ───────────────────────────────────────
_client = None

def _get_client():
    global _client
    if _client is None:
        http_client = httpx.Client(
            proxy=PROXY_URL if PROXY_URL else None,
            timeout=httpx.Timeout(3600.0),
        )
        _client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            http_client=http_client,
        )
    return _client


# ── Промпты ──────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a strict grader for a RAG QA competition.
Your task: decide if the participant's answer is correct or wrong compared to the gold answer.

You MUST make a definitive choice — no partial credit exists.
If the answer captures the key facts and meaning, mark it correct.
If it is incomplete, vague, or wrong — mark it wrong.

Respond ONLY with a valid JSON object and nothing else.
Format: {"score": 0|1}

Scoring rules:
  1 — correct: semantically equivalent to the gold answer, key facts match
  0 — wrong: missing key facts, incorrect, empty, or irrelevant
"""

USER_PROMPT_TEMPLATE = """\
Question:
{question}

Gold answer:
{gold}

Participant answer:
{pred}
"""


def _parse_score(text: str) -> int:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return 0
    try:
        obj = json.loads(m.group(0))
        s = int(obj.get("score", 0))
        return 1 if s == 1 else 0
    except Exception:
        return 0


def _eval_one(qid: str, question: str, gold: str, pred: str) -> dict:
    pred = (pred or "").strip()
    if not pred:
        return {"id": qid, "question": question, "gold": gold, "pred": pred, "score": 0}

    prompt = USER_PROMPT_TEMPLATE.format(question=question, gold=gold, pred=pred)
    try:
        resp = _get_client().chat.completions.create(
            model=EVAL_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        score = _parse_score(resp.choices[0].message.content)
    except Exception as e:
        logger.error("[judge] error on %s: %s", qid, e)
        score = 0

    return {"id": qid, "question": question, "gold": gold, "pred": pred, "score": score}


def evaluate_submission(submit_path: str) -> dict:
    sub_rows = load_jsonl(submit_path)
    pred_map = {str(x["id"]): str(x.get("answer", "")).strip() for x in sub_rows}

    gold_rows = load_jsonl(get_gold_path())

    gold_map = {}
    question_map = {}
    for x in gold_rows:
        xid = str(x["id"])
        gold_map[xid] = str(x.get("answer", ""))
        if "question" in x:
            question_map[xid] = x["question"]

    try:
        pub_questions = load_jsonl(QUESTIONS_PATH)
        for q in pub_questions:
            qid = str(q["id"])
            if qid not in question_map:
                question_map[qid] = q.get("question", "")
    except Exception:
        pass

    total = len(gold_map)
    answered_ids = [qid for qid in gold_map if pred_map.get(qid, "")]

    details = []
    with ThreadPoolExecutor(max_workers=EVAL_CONCURRENCY) as executor:
        futures = {
            executor.submit(
                _eval_one,
                qid,
                question_map.get(qid, ""),
                gold_map[qid],
                pred_map[qid],
            ): qid
            for qid in answered_ids
        }
        for future in as_completed(futures):
            try:
                details.append(future.result())
            except Exception as e:
                qid = futures[future]
                logger.error("[judge] future error on %s: %s", qid, e)
                details.append({"id": qid, "score": 0})

    scores = [d["score"] for d in details]
    return {
        "zeros": scores.count(0),
        "ones": scores.count(1),
        "n": len(answered_ids),
        "total": total,
        "details": details,
    }
