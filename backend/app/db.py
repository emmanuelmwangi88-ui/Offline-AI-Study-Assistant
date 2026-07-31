"""
db.py
-----
Lightweight SQLite persistence for student quiz attempts and progress
tracking. No ORM — kept simple and explicit since the scope here is small
(single-laptop, solo-student demo).
"""

import sqlite3
from pathlib import Path
from typing import List, Dict
from collections import Counter

DB_PATH = Path(__file__).resolve().parent.parent / "study_assistant.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(cur, table: str, column: str, column_def: str):
    """
    Adds a column to an existing table if it's missing. CREATE TABLE IF NOT
    EXISTS only creates a table the first time — if the table already exists
    from before a schema change (e.g. an older version of this app), new
    columns silently never get added, and every query referencing them fails
    with 'no such column'. This makes schema changes safe on an existing
    database file instead of requiring people to delete it.
    """
    cur.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in cur.fetchall()}
    if column not in existing_columns:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            correct INTEGER NOT NULL,  -- 0 or 1
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            sources TEXT,        -- comma-separated source filenames
            model_used TEXT,
            grounded INTEGER DEFAULT 1,  -- 0 if answered without ingested notes
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students (id)
        )
    """)

    # Migration safety net: if chat_messages already existed from before the
    # `grounded` column was added, add it now instead of failing at query time.
    _ensure_column(cur, "chat_messages", "grounded", "INTEGER DEFAULT 1")

    conn.commit()
    conn.close()


def ensure_student(student_id: str, name: str = ""):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO students (id, name) VALUES (?, ?)",
        (student_id, name),
    )
    conn.commit()
    conn.close()


def log_attempt(student_id: str, topic: str, correct: bool):
    ensure_student(student_id)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO attempts (student_id, topic, correct) VALUES (?, ?, ?)",
        (student_id, topic, int(correct)),
    )
    conn.commit()
    conn.close()


def get_attempts(student_id: str) -> List[Dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT topic, correct, timestamp FROM attempts WHERE student_id = ? ORDER BY timestamp",
        (student_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_topic_priority(student_id: str) -> List[Dict]:
    """
    Spaced-repetition-style priority score per topic, combining:
    - accuracy (lower accuracy = higher priority)
    - recency (longer since last reviewed = higher priority)

    score = 0.7 * (1 - accuracy) + 0.3 * recency_factor
    where recency_factor approaches 1 the longer it's been since the last
    attempt (using a simple days/(days+3) curve so a few days out already
    matters, but keeps climbing for topics left much longer).

    Returns topics sorted highest-priority first. This is a simple heuristic,
    not real spaced-repetition scheduling (e.g. SM-2) — a good "next step"
    to note in the notebook if you want to go further.
    """
    from datetime import datetime

    attempts = get_attempts(student_id)
    if not attempts:
        return []

    by_topic: Dict[str, List[Dict]] = {}
    for a in attempts:
        by_topic.setdefault(a["topic"], []).append(a)

    now = datetime.now()
    results = []
    for topic, records in by_topic.items():
        correct_list = [r["correct"] for r in records]
        accuracy = sum(correct_list) / len(correct_list)

        last_timestamp = max(r["timestamp"] for r in records)
        try:
            last_dt = datetime.strptime(last_timestamp, "%Y-%m-%d %H:%M:%S")
            days_since = (now - last_dt).total_seconds() / 86400
        except ValueError:
            days_since = 0

        recency_factor = days_since / (days_since + 3)
        score = 0.7 * (1 - accuracy) + 0.3 * recency_factor

        results.append({
            "topic": topic,
            "accuracy": round(accuracy, 2),
            "days_since_last_review": round(days_since, 2),
            "priority_score": round(score, 3),
        })

    results.sort(key=lambda r: r["priority_score"], reverse=True)
    return results


def get_weak_topics(student_id: str, min_attempts: int = 1) -> List[str]:
    """
    A topic counts as 'weak' if the student's accuracy on it is below 60%,
    based on their most recent attempts. Simple heuristic, easy to tune
    and easy to explain/justify in the notebook.
    """
    attempts = get_attempts(student_id)
    if not attempts:
        return []

    by_topic: Dict[str, List[int]] = {}
    for a in attempts:
        by_topic.setdefault(a["topic"], []).append(a["correct"])

    weak = []
    for topic, results in by_topic.items():
        if len(results) < min_attempts:
            continue
        accuracy = sum(results) / len(results)
        if accuracy < 0.6:
            weak.append(topic)

    return weak


def get_topic_accuracy_summary(student_id: str) -> Dict[str, float]:
    attempts = get_attempts(student_id)
    by_topic: Dict[str, List[int]] = {}
    for a in attempts:
        by_topic.setdefault(a["topic"], []).append(a["correct"])

    return {
        topic: round(sum(results) / len(results), 2)
        for topic, results in by_topic.items()
    }


def log_chat_message(student_id: str, question: str, answer: str,
                      sources: List[str] = None, model_used: str = None,
                      grounded: bool = True):
    ensure_student(student_id)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO chat_messages (student_id, question, answer, sources, model_used, grounded)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (student_id, question, answer, ",".join(sources or []), model_used, int(grounded)),
    )
    conn.commit()
    conn.close()


def get_chat_history(student_id: str, limit: int = 50) -> List[Dict]:
    """Returns messages oldest-first, capped at `limit` most recent turns."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT question, answer, sources, model_used, grounded, timestamp
           FROM chat_messages WHERE student_id = ?
           ORDER BY id DESC LIMIT ?""",
        (student_id, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    rows.reverse()  # oldest-first for natural chat display
    for r in rows:
        r["sources"] = r["sources"].split(",") if r["sources"] else []
        r["grounded"] = bool(r["grounded"])
    return rows


def clear_chat_history(student_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM chat_messages WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # quick manual sanity check
    init_db()
    log_attempt("student_1", "photosynthesis", True)
    log_attempt("student_1", "photosynthesis", False)
    log_attempt("student_1", "cell_division", False)

    print("Attempts:", get_attempts("student_1"))
    print("Weak topics:", get_weak_topics("student_1"))
    print("Accuracy summary:", get_topic_accuracy_summary("student_1"))
