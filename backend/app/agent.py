"""
agent.py
--------
The agentic layer on top of the RAG core. Tools the assistant can invoke,
plus a simple router that decides which tool (if any) a user's message
should trigger.

Tools:
    generate_quiz(index, topics, num_questions, difficulty) -> quiz grounded in notes
    grade_answer(user_answer, expected_answer)              -> fuzzy correctness check
    track_weak_topics(student_id, topic, correct)           -> logs a quiz attempt
    suggest_next_topic(student_id)                           -> recommends what to study next
    summarize_session(student_id)                            -> recap of a study session

This is intentionally simple (manual keyword routing rather than native
function-calling) so it's easy to explain and justify in the notebook.
Swap in real function-calling later if your chosen Ollama model supports it.
"""

import json
import re
import difflib
from typing import List, Dict, Optional, Union

from app.rag import StudyIndex, call_ollama, OLLAMA_MODEL, format_chat_history
from app import db


# ---------------------------------------------------------------------------
# Tool 1: generate_quiz
# ---------------------------------------------------------------------------

def _chunks_for_topics(index: StudyIndex, topics: List[str], max_chunks: int = 8) -> List[str]:
    """Pull chunks tagged with any of the given topics from the index metadata
    directly, rather than via similarity search, since we want topic-scoped
    material, not query-scoped material."""
    matches = [m["text"] for m in index.metadata if m.get("topic") in topics]
    return matches[:max_chunks] if matches else [m["text"] for m in index.metadata[:max_chunks]]


DIFFICULTY_INSTRUCTIONS = {
    "easy": "Keep questions simple and direct, testing basic recall of facts stated in the notes.",
    "medium": "Ask questions that require understanding a concept, not just recalling a single fact.",
    "hard": "Ask questions that require connecting multiple ideas from the notes or applying a concept "
            "to a new situation, not just recalling what's written.",
}


def generate_quiz(index: StudyIndex, topics: Union[str, List[str]], num_questions: int = 3,
                   difficulty: str = "medium") -> Dict:
    """
    topics: a single topic string, or a list of topics for a mixed-review quiz.
    difficulty: "easy", "medium", or "hard" — adjusts the prompt's instruction
    to the model; note this is a prompting technique, not a verified guarantee
    the model's actual output difficulty varies meaningfully — worth checking
    in your notebook's evaluation section with real output.
    """
    topic_list = [topics] if isinstance(topics, str) else list(topics)
    topic_label = ", ".join(topic_list)
    difficulty_instruction = DIFFICULTY_INSTRUCTIONS.get(difficulty, DIFFICULTY_INSTRUCTIONS["medium"])

    context_chunks = _chunks_for_topics(index, topic_list)
    grounded = bool(context_chunks)

    if grounded:
        context_text = "\n\n".join(context_chunks)
        prompt = f"""Based ONLY on the study notes below, write {num_questions} short-answer
quiz questions to test a student's understanding of: {topic_label}.
Difficulty level: {difficulty}. {difficulty_instruction}

Notes:
{context_text}

Respond ONLY with a JSON array, no other text, in this exact format:
[
  {{"question": "...", "expected_answer": "..."}},
  ...
]"""
    else:
        # No notes ingested for these topics — fall back to general knowledge.
        # Not grounded RAG; callers should surface this to the user.
        prompt = f"""No study notes are available yet. Using your own general
knowledge, write {num_questions} short-answer quiz questions about: {topic_label}.
Difficulty level: {difficulty}. {difficulty_instruction}

Respond ONLY with a JSON array, no other text, in this exact format:
[
  {{"question": "...", "expected_answer": "..."}},
  ...
]"""

    # Quiz JSON needs more room than the default cap as num_questions grows —
    # roughly 60 tokens/question is generous for a short-answer Q&A pair.
    quiz_num_predict = max(350, num_questions * 60)

    raw = call_ollama(prompt, model=OLLAMA_MODEL, num_predict=quiz_num_predict)
    questions = _parse_quiz_json(raw)

    if questions is None:
        # Small local models occasionally ignore the "JSON only" instruction.
        # One retry with a stricter, more explicit prompt is cheap and often
        # fixes it before we fall back to the raw-text placeholder.
        retry_prompt = f"""Your previous response was not valid JSON. Respond with
ONLY a JSON array, nothing else — no explanation, no markdown fences, no
leading or trailing text. Example of the exact format required:
[{{"question": "What is X?", "expected_answer": "Y"}}]

Now write {num_questions} short-answer quiz questions about "{topic_label}" in
exactly that format."""
        raw_retry = call_ollama(retry_prompt, model=OLLAMA_MODEL, num_predict=quiz_num_predict)
        questions = _parse_quiz_json(raw_retry)

    if questions is None:
        # Both attempts failed — fall back to raw text as a single question
        # rather than crashing. Worth revisiting in the notebook's
        # "limitations" section if this happens often with your chosen model.
        questions = [{"question": raw, "expected_answer": ""}]

    return {"questions": questions, "grounded": grounded}


def _parse_quiz_json(raw: str):
    """Best-effort JSON parse of a quiz response, tolerant of markdown fences.
    Returns None (not an exception) on failure so callers can decide to retry."""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Answer grading (fuzzy, not exact-string-match)
# ---------------------------------------------------------------------------

GRADE_SIMILARITY_THRESHOLD = 0.6  # tuned loosely; see notebook for discussion


def grade_answer(user_answer: str, expected_answer: str) -> Dict:
    """
    Grades a quiz answer with fuzzy string matching instead of requiring an
    exact match, since students paraphrase, misspell, or add/drop small
    words ("cell division" vs "Cell Division." vs "division of cells").
    Uses difflib's SequenceMatcher ratio on normalized (lowercased,
    whitespace-collapsed) text. Not semantic — "wrong answer that happens to
    share words" can still slip through, and a correct answer phrased very
    differently from expected_answer can still be marked wrong. A model-based
    grader (asking the LLM "is this correct?") would be more accurate but
    costs a model call per answer; this stays fast and free. Worth comparing
    both approaches in the notebook if you have time.
    """
    def normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    a = normalize(user_answer)
    b = normalize(expected_answer)

    if not a or not b:
        return {"correct": False, "similarity": 0.0}

    if a == b:
        return {"correct": True, "similarity": 1.0}

    similarity = difflib.SequenceMatcher(None, a, b).ratio()

    # Also treat it as correct if the user's answer contains the expected
    # answer (or vice versa) as a substring — catches "mitosis" matching
    # "cell division via mitosis" style short-vs-long answers.
    substring_match = a in b or b in a

    correct = similarity >= GRADE_SIMILARITY_THRESHOLD or substring_match
    return {"correct": correct, "similarity": round(similarity, 3)}


# ---------------------------------------------------------------------------
# Tool 2: track_weak_topics
# ---------------------------------------------------------------------------

def track_weak_topics(student_id: str, topic: str, correct: bool) -> None:
    db.log_attempt(student_id, topic, correct)


# ---------------------------------------------------------------------------
# Tool 3: suggest_next_topic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Study session summary
# ---------------------------------------------------------------------------

def summarize_session(student_id: str, chat_turns: int = 10) -> Dict:
    """
    Generates a short recap of recent activity: what topics came up in
    conversation, and how quiz performance looked. Combines chat history and
    quiz attempts, then asks the model to synthesize a brief summary — this
    is a genuinely agentic step (synthesis, not retrieval) rather than RAG.
    """
    history = db.get_chat_history(student_id, limit=chat_turns)
    priorities = db.get_topic_priority(student_id)

    if not history and not priorities:
        return {
            "summary": "No activity yet this session — ask a question or take a quiz to get started.",
            "topics_covered": [],
        }

    topics_covered = sorted({t["topic"] for t in priorities}) if priorities else []
    history_text = format_chat_history(history)

    quiz_text = ""
    if priorities:
        quiz_lines = [
            f"- {p['topic']}: {p['accuracy'] * 100:.0f}% accuracy" for p in priorities
        ]
        quiz_text = "Quiz performance by topic:\n" + "\n".join(quiz_lines)

    prompt = f"""Summarize this student's study session in 2-4 short sentences.
Mention what topics they covered and how they're doing, in an encouraging
but honest tone. Don't invent details not present below.

Recent conversation:
{history_text if history_text else "(no questions asked this session)"}

{quiz_text if quiz_text else "(no quizzes taken this session)"}

Summary:"""

    summary = call_ollama(prompt, model=OLLAMA_MODEL, num_predict=200)

    return {"summary": summary, "topics_covered": topics_covered}


def export_study_summary(student_id: str) -> str:
    """
    Builds a markdown study summary: topics covered, per-topic accuracy, and
    what's flagged as due for review. Pure data formatting — no model call,
    so it's instant and works even without Ollama running.
    """
    from datetime import datetime

    priorities = db.get_topic_priority(student_id)
    history = db.get_chat_history(student_id, limit=1000)

    lines = [
        f"# Study Summary — {student_id}",
        f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
    ]

    if not priorities and not history:
        lines.append("No activity recorded yet.")
        return "\n".join(lines)

    lines.append(f"**Questions asked:** {len(history)}")
    lines.append(f"**Topics quizzed:** {len(priorities)}")
    lines.append("")

    if priorities:
        lines.append("## Topic performance")
        lines.append("")
        lines.append("| Topic | Accuracy | Days since last review | Priority |")
        lines.append("|---|---|---|---|")
        for p in priorities:
            flag = "⚠️ Review soon" if p["priority_score"] > 0.5 else "✅ On track"
            lines.append(
                f"| {p['topic']} | {p['accuracy'] * 100:.0f}% | "
                f"{p['days_since_last_review']:.0f} | {flag} |"
            )
        lines.append("")

    if history:
        lines.append("## Recent questions")
        lines.append("")
        for turn in history[-10:]:
            lines.append(f"- **Q:** {turn['question']}")
        lines.append("")

    return "\n".join(lines)


def suggest_next_topic(student_id: str) -> Dict:
    priorities = db.get_topic_priority(student_id)

    if not priorities:
        return {
            "suggested_topic": None,
            "reason": "No quiz history yet — take a quiz first so I can "
                      "track your progress across topics.",
        }

    top = priorities[0]
    days = top["days_since_last_review"]
    recency_note = (
        f"and it's been {days:.0f} day(s) since you reviewed it" if days >= 1
        else "and you reviewed it very recently, so it's due for reinforcement"
    )

    return {
        "suggested_topic": top["topic"],
        "reason": (
            f"Your accuracy on '{top['topic']}' is {top['accuracy'] * 100:.0f}%, "
            f"{recency_note}. This combines how well you're doing with how "
            f"long it's been since you last practiced it."
        ),
    }


# ---------------------------------------------------------------------------
# Simple router: decide whether a user message should trigger a tool
# ---------------------------------------------------------------------------

QUIZ_TRIGGERS = ["quiz me", "test me", "give me questions", "make a quiz"]
PROGRESS_TRIGGERS = ["what should i study", "what's next", "suggest a topic", "my progress"]


def route_message(message: str) -> Optional[str]:
    """
    Returns one of: "quiz", "progress", or None (meaning: treat as a plain
    RAG question). This is deliberately simple keyword matching — document
    this choice and its tradeoffs in the notebook's agent-implementation
    section.
    """
    lower = message.lower()

    if any(trigger in lower for trigger in QUIZ_TRIGGERS):
        return "quiz"
    if any(trigger in lower for trigger in PROGRESS_TRIGGERS):
        return "progress"
    return None


if __name__ == "__main__":
    # Quick manual sanity check for the non-LLM parts (routing + tracking)
    db.init_db()

    assert route_message("Can you quiz me on photosynthesis?") == "quiz"
    assert route_message("What should I study next?") == "progress"
    assert route_message("What is photosynthesis?") is None
    print("Routing checks passed.")

    track_weak_topics("student_1", "photosynthesis", False)
    track_weak_topics("student_1", "photosynthesis", False)
    print(suggest_next_topic("student_1"))
