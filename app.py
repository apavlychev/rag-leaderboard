# app.py — RAG Leaderboard v2.1 (local deploy, LLM-as-judge via OpenAI-compatible API)
import os
import json
import logging
import time
import pandas as pd
import gradio as gr
from pathlib import Path

from src.submission.check_validity import check_submission
from src.submission.submit import evaluate_submission
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

LEADERBOARD_PATH = "data/results/leaderboard.csv"
DETAILS_PATH = "data/results/eval_details.jsonl"

# Файлы для скачивания
QUESTIONS_DOWNLOAD_PATH = os.getenv("QUESTIONS_DOWNLOAD_PATH", QUESTIONS_PATH)
DOCUMENTS_ZIP_PATH = os.getenv("DOCUMENTS_ZIP_PATH", "data/documents.zip")

SHOW_DETAILS = os.getenv("SHOW_DETAILS", "0") == "1"

LB_COLUMNS = [
    "filename",
    "Wrong", "Correct",
    "accuracy", "n", "total", "eval_time", "timestamp",
]

LB_DISPLAY_COLUMNS = LB_COLUMNS

# ── Logging ─────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Leaderboard ───────────────────────────────────────────────────────────────

def ensure_leaderboard():
    if not os.path.exists(LEADERBOARD_PATH):
        pd.DataFrame(columns=LB_COLUMNS).to_csv(LEADERBOARD_PATH, index=False)
        return
    df = pd.read_csv(LEADERBOARD_PATH)
    changed = False
    for col in LB_COLUMNS:
        if col not in df.columns:
            df[col] = ""
            changed = True
    extra = [c for c in df.columns if c not in LB_COLUMNS]
    if extra:
        df = df.drop(columns=extra)
        changed = True
    if changed:
        df.to_csv(LEADERBOARD_PATH, index=False)


def sort_leaderboard(df):
    return df.sort_values(
        by=["accuracy", "Correct"],
        ascending=[False, False],
    ).reset_index(drop=True)


def load_sorted_leaderboard():
    ensure_leaderboard()
    df = pd.read_csv(LEADERBOARD_PATH)
    if df.empty:
        return df
    df = sort_leaderboard(df).reset_index(drop=False)
    df["Place"] = df["index"] + 1
    return df[["Place"] + LB_DISPLAY_COLUMNS]


# ── Eval details ──────────────────────────────────────────────────────────────

def load_all_detail_records() -> list[dict]:
    if not os.path.exists(DETAILS_PATH):
        return []
    records = []
    with open(DETAILS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def save_detail_record(filename: str, timestamp: str, details: list) -> None:
    record = {"filename": filename, "timestamp": timestamp, "details": details}
    with open(DETAILS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def list_submissions() -> list[str]:
    records = load_all_detail_records()
    return [f"{r['timestamp']} — {r['filename']}" for r in reversed(records)]


def format_details_html(details: list) -> str:
    if not details:
        return "<p>No evaluation details available.</p>"

    groups = {0: [], 1: []}
    for d in details:
        score = d.get("score", 0)
        score = 1 if score >= 1 else 0
        groups[score].append(d)

    labels = {
        0: ("🔴 Wrong", "#ffeaea"),
        1: ("🟢 Correct", "#eaffea"),
    }

    html = ""
    for score in [1, 0]:
        items = groups[score]
        if not items:
            continue
        title, bg = labels[score]
        html += f"<h3>{title} ({len(items)})</h3>"
        for d in items:
            qid = d.get("id", "")
            question = d.get("question", "")
            gold = d.get("gold", "")
            pred = d.get("pred", "")
            html += f"""
<div style="background:{bg};border-radius:8px;padding:12px;margin-bottom:10px;font-size:13px;">
  <b>[{qid}]</b> {question}<br><br>
  <b>Gold:</b> {gold}<br><br>
  <b>Pred:</b> {pred}
</div>"""
    return html


def load_details_by_label(label: str) -> str:
    if not label:
        return "<p>Select a submission above.</p>"
    records = load_all_detail_records()
    for r in reversed(records):
        if f"{r['timestamp']} — {r['filename']}" == label:
            return format_details_html(r.get("details", []))
    return "<p>Submission not found.</p>"


def load_latest_details_html() -> str:
    records = load_all_detail_records()
    if not records:
        return "<p>No evaluation details yet.</p>"
    return format_details_html(records[-1].get("details", []))


# ── Downloads ─────────────────────────────────────────────────────────────────

def download_questions():
    if os.path.exists(QUESTIONS_DOWNLOAD_PATH):
        return QUESTIONS_DOWNLOAD_PATH
    return None


def download_documents():
    if os.path.exists(DOCUMENTS_ZIP_PATH):
        return DOCUMENTS_ZIP_PATH
    return None


# ── Submit ────────────────────────────────────────────────────────────────────

def submit_file(file_obj):
    ensure_leaderboard()

    no_details = "<p>No details.</p>"

    if file_obj is None:
        return "❌ Please upload a JSONL file", load_sorted_leaderboard(), gr.update(choices=list_submissions()), no_details

    file_path = file_obj.name
    filename = Path(file_path).name

    ok, msg = check_submission(file_path, QUESTIONS_PATH)
    if not ok:
        return f"❌ Invalid submission: {msg}", load_sorted_leaderboard(), gr.update(choices=list_submissions()), no_details

    t_start = time.time()
    try:
        result = evaluate_submission(file_path)
    except Exception as e:
        return f"❌ Evaluation failed: {e}", load_sorted_leaderboard(), gr.update(choices=list_submissions()), no_details
    eval_time = round(time.time() - t_start, 1)

    n = result["n"]
    total = result["total"]
    correct = result["ones"]
    wrong = result["zeros"]
    accuracy = round(correct / max(n, 1), 4)
    details = result.get("details", [])
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

    try:
        save_detail_record(filename, timestamp, details)
    except Exception as e:
        logger.warning("Could not save details: %s", e)

    row = {
        "timestamp": timestamp, "filename": filename,
        "Wrong": wrong, "Correct": correct,
        "accuracy": accuracy,
        "n": n, "total": total, "eval_time": f"{eval_time}s",
    }
    df = pd.read_csv(LEADERBOARD_PATH)
    df.loc[len(df)] = row
    df.to_csv(LEADERBOARD_PATH, index=False)

    summary = (
        f"✅ Submitted! "
        f"Answered: {n}/{total} | Wrong: {wrong} | Correct: {correct} | "
        f"Accuracy: {accuracy:.1%} | Time: {eval_time}s"
    )
    choices = list_submissions()
    new_label = f"{timestamp} — {filename}"
    return summary, load_sorted_leaderboard(), gr.update(choices=choices, value=new_label), format_details_html(details)


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui():
    ensure_leaderboard()

    with gr.Blocks(title="RAG Arena") as demo:

        gr.Markdown(
            "# 🏁 RAG Arena — LLM-as-Judge Benchmark\n"
            "Upload your system's answers in JSONL format to see how they score. "
            "Each answer is graded by an **LLM-as-judge** as **Correct ✅ or Wrong ❌**."
        )

        # ── 1. Лидерборд ──────────────────────────────────────────────────────
        gr.Markdown("## 📊 Leaderboard")
        out_df = gr.Dataframe(value=load_sorted_leaderboard(), interactive=False, wrap=True, label="")
        refresh_btn = gr.Button("🔄 Refresh", variant="secondary")

        gr.Markdown("---")

        # ── 2. Скачать датасет ────────────────────────────────────────────────
        gr.Markdown("## 📥 Download dataset")
        with gr.Row():
            btn_questions = gr.DownloadButton(
                label="📋 Download questions (JSONL)",
                value=QUESTIONS_DOWNLOAD_PATH if os.path.exists(QUESTIONS_DOWNLOAD_PATH) else None,
                variant="secondary",
            )
            btn_documents = gr.DownloadButton(
                label="📄 Download documents (ZIP)",
                value=DOCUMENTS_ZIP_PATH if os.path.exists(DOCUMENTS_ZIP_PATH) else None,
                variant="secondary",
            )

        gr.Markdown("---")

        # ── 3. Форма сабмита ───────────────────────────────────────────────────
        gr.Markdown(
            "## 📤 Submit your answers\n\n"
            "**Format** — one JSON per line:\n"
            "```json\n"
            "{\"id\": \"0\", \"answer\": \"Your answer here\"}\n"
            "```\n"
            "`id` must match the question IDs from the public question set."
        )
        file_in = gr.File(label="Upload JSONL (answers)", file_types=[".jsonl"])
        submit_btn = gr.Button("Submit", variant="primary")
        out_msg = gr.Markdown()

        gr.Markdown("---")

        # ── 4. Dataset info ────────────────────────────────────────────────────
        gr.Markdown(
            "## 📋 Dataset info\n"
            f"- Judge model: {OPENAI_BASE_URL}/?model={EVAL_MODEL} (via OpenAI API compatibility, `grok-4-1-fast-reasoning` by default)\n"
            "- Scoring: **binary** — Correct or Wrong, no partial credit\n"
            f"- Gold answers: stored privately on server {get_gold_path()}, loaded at evaluation time\n"
        )

        gr.Markdown("---")

        # ── 5. Детали оценки ───────────────────────────────────────────────────
        if SHOW_DETAILS:
            gr.Markdown("## 🔍 Evaluation details")
            details_dropdown = gr.Dropdown(
                choices=list_submissions(),
                value=list_submissions()[0] if list_submissions() else None,
                label="Select submission",
                interactive=True,
            )
            out_details = gr.HTML(value=load_latest_details_html())
        else:
            gr.Markdown("## 🔍 Evaluation details OFF")
            details_dropdown = gr.Dropdown(visible=False)
            out_details = gr.HTML(visible=False)

        # ── Привязка событий ───────────────────────────────────────────────────
        def do_refresh():
            ensure_leaderboard()
            subs = list_submissions()
            if SHOW_DETAILS:
                return (
                    load_sorted_leaderboard(),
                    gr.update(choices=subs, value=subs[0] if subs else None),
                    load_latest_details_html(),
                )
            return load_sorted_leaderboard(), gr.update(), gr.update()

        refresh_btn.click(
            fn=do_refresh,
            inputs=[],
            outputs=[out_df, details_dropdown, out_details],
        )
        if SHOW_DETAILS:
            details_dropdown.change(
                fn=load_details_by_label,
                inputs=[details_dropdown],
                outputs=[out_details],
            )
        submit_btn.click(
            fn=submit_file,
            inputs=[file_in],
            outputs=[out_msg, out_df, details_dropdown, out_details],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
