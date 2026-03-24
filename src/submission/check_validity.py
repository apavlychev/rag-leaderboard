import json
from src.envs import load_jsonl


def check_submission(path: str, questions_path: str) -> tuple[bool, str]:
    """
    Проверяет корректность JSONL-файла с ответами.
    Возвращает (ok: bool, message: str).
    """
    # Загружаем вопросы, чтобы проверить что все id присутствуют
    try:
        questions = load_jsonl(questions_path)
        expected_ids = {str(q["id"]) for q in questions}
    except Exception as e:
        return False, f"Cannot load questions: {e}"

    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except Exception as e:
        return False, f"Cannot read file: {e}"

    if not lines:
        return False, "File is empty"

    found_ids = set()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except Exception:
            return False, f"Line {i+1} is not valid JSON"

        if "id" not in item:
            return False, f"Missing 'id' in line {i+1}"
        if "answer" not in item:
            return False, f"Missing 'answer' in line {i+1}"

        found_ids.add(str(item["id"]))

    # Предупреждаем об отсутствующих вопросах (не блокируем — система засчитает 0)
    missing = expected_ids - found_ids
    extra = found_ids - expected_ids

    msg_parts = [f"{len(lines)} lines parsed"]
    if missing:
        msg_parts.append(f"⚠️ {len(missing)} question IDs missing (will score 0)")
    if extra:
        msg_parts.append(f"⚠️ {len(extra)} unknown IDs will be ignored")

    return True, " | ".join(msg_parts)
