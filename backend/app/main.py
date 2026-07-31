"""
main.py
-------
FastAPI backend exposing the study assistant's AI functionality.

Run with (from inside backend/):
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI, UploadFile, Form, HTTPException, File, Response
from pathlib import Path
import shutil
import tempfile

from app.rag import (
    StudyIndex, answer_question, answer_general, answer_with_attachment,
    load_document, format_chat_history, DATA_RAW_DIR,
)
from app import agent, db
from app.models import (
    AskRequest, AskResponse,
    QuizRequest, QuizResponse,
    GradeRequest, GradeResponse,
    AttemptRequest,
    ProgressResponse,
    SuggestionResponse,
    SessionSummaryResponse,
    NotesListResponse,
    ChatHistoryResponse,
)

app = FastAPI(title="Offline Study Assistant API")

# One shared in-memory index for this simple, single-student demo.
# For multiple concurrent students you'd key indexes by student_id instead.
study_index = StudyIndex()
_index_built = False

CHAT_HISTORY_TURNS = 6  # how many recent Q&A turns to feed back as context


def _recent_history_text(student_id: str) -> str:
    recent = db.get_chat_history(student_id, limit=CHAT_HISTORY_TURNS)
    return format_chat_history(recent)


@app.on_event("startup")
def startup():
    global _index_built
    db.init_db()
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Reload a previously saved index so restarting the backend doesn't
    # throw away everything you've already ingested (and re-embedding
    # everything from scratch would be slow anyway).
    try:
        study_index.load()
        _index_built = True
    except (FileNotFoundError, OSError):
        pass  # nothing saved yet — normal on first run


@app.post("/ingest")
async def ingest(file: UploadFile, topic: str = Form("general")):
    """
    Accepts .txt, .pdf, or image files (.png/.jpg/.jpeg/.webp — photographed
    notes). Images are transcribed/described via the local vision model
    (moondream) in load_document() before being chunked and embedded like
    any other text source.
    """
    global _index_built

    save_path = DATA_RAW_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    text = load_document(save_path)
    if not text.strip():
        raise HTTPException(400, "Could not extract any text from the uploaded file.")

    ocr_warning = None
    if save_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and len(text.strip()) < 20:
        # The vision model returned very little — likely a blurry photo,
        # low light, or handwriting it couldn't read. Still proceed (some
        # notes really are short), but flag it so the user can retake the photo.
        ocr_warning = (
            "The extracted text was very short — the image may be blurry, "
            "poorly lit, or hard to read. Consider retaking the photo if the "
            "answers seem off."
        )

    documents = [{"text": text, "source": file.filename, "topic": topic}]

    # Incremental add: only the new document gets chunked/embedded, so
    # ingestion time no longer grows with how much you've already uploaded.
    study_index.add(documents)
    _index_built = True

    study_index.save()
    response = {"status": "ok", "filename": file.filename, "topic": topic}
    if ocr_warning:
        response["warning"] = ocr_warning
    return response


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    # Agentic routing: check if this "question" is actually a request
    # to be quizzed or for a study suggestion instead of a plain RAG lookup.
    route = agent.route_message(req.question)
    if route == "quiz":
        return AskResponse(
            answer="Looks like you want a quiz — use the Quiz tab to generate one.",
            sources=[], grounded=True,
        )
    if route == "progress":
        suggestion = agent.suggest_next_topic(req.student_id)
        return AskResponse(answer=suggestion["reason"], sources=[], grounded=True)

    history_text = _recent_history_text(req.student_id)

    if _index_built:
        result = answer_question(study_index, req.question, chat_history=history_text)
        grounded = True
    else:
        # No notes ingested yet — fall back to a general-knowledge answer
        # rather than blocking the user. Clearly marked as ungrounded so the
        # frontend can warn that this isn't based on the student's own notes.
        result = answer_general(req.question, chat_history=history_text)
        grounded = False

    db.log_chat_message(
        req.student_id, req.question, result["answer"],
        sources=result["sources"], model_used=result.get("model_used"),
        grounded=grounded,
    )
    return AskResponse(
        answer=result["answer"],
        sources=result["sources"],
        model_used=result.get("model_used"),
        grounded=grounded,
    )


@app.post("/ask_with_file", response_model=AskResponse)
async def ask_with_file(
    question: str = Form(...),
    student_id: str = Form("student_1"),
    file: UploadFile = File(...),
):
    """
    Answers a question about a file/image attached directly to a chat
    message. Unlike /ingest, this does NOT permanently add the file to the
    study index — it's a one-off attachment for this question only.
    """
    suffix = Path(file.filename).suffix or ""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        attachment_text = load_document(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not attachment_text.strip():
        raise HTTPException(400, "Could not extract any content from the attached file.")

    history_text = _recent_history_text(student_id)
    result = answer_with_attachment(study_index, question, attachment_text, chat_history=history_text)

    db.log_chat_message(
        student_id, question, result["answer"],
        sources=[file.filename] + result["sources"],
        model_used=result.get("model_used"),
        grounded=True,  # grounded in the attached file, even if not in the permanent index
    )
    return AskResponse(
        answer=result["answer"],
        sources=[file.filename] + result["sources"],
        model_used=result.get("model_used"),
        grounded=True,
    )


@app.post("/quiz", response_model=QuizResponse)
async def quiz(req: QuizRequest):
    result = agent.generate_quiz(
        study_index, req.resolved_topics(), req.num_questions, req.difficulty
    )
    return QuizResponse(questions=result["questions"], grounded=result["grounded"])


@app.post("/grade", response_model=GradeResponse)
async def grade(req: GradeRequest):
    """
    Fuzzy grading for a quiz answer — no exact-match requirement, so
    paraphrases/typos/case differences aren't unfairly marked wrong.
    No model call, so this is instant.
    """
    result = agent.grade_answer(req.user_answer, req.expected_answer)
    return GradeResponse(**result)


@app.get("/notes", response_model=NotesListResponse)
async def list_notes():
    """Lists currently-ingested source filenames, for a 'manage notes' UI."""
    return NotesListResponse(sources=study_index.list_sources())


@app.delete("/notes/{filename}")
async def delete_note(filename: str):
    global _index_built
    removed = study_index.remove_source(filename)
    if removed == 0:
        raise HTTPException(404, f"No ingested chunks found for '{filename}'.")

    if study_index.embeddings is None:
        _index_built = False

    study_index.save()
    return {"status": "deleted", "filename": filename, "chunks_removed": removed}


@app.delete("/notes")
async def clear_all_notes():
    """Clears every ingested note (a fresh start), separate from clearing chat history."""
    global _index_built
    study_index.embeddings = None
    study_index.metadata = []
    _index_built = False
    study_index.save()
    return {"status": "cleared"}


@app.get("/session_summary/{student_id}", response_model=SessionSummaryResponse)
async def session_summary(student_id: str):
    result = agent.summarize_session(student_id)
    return SessionSummaryResponse(**result)


@app.get("/export/{student_id}")
async def export_summary(student_id: str):
    """Returns a downloadable markdown study summary (progress + recent questions)."""
    markdown_text = agent.export_study_summary(student_id)
    return Response(
        content=markdown_text,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{student_id}_study_summary.md"'},
    )


@app.post("/attempt")
async def attempt(req: AttemptRequest):
    agent.track_weak_topics(req.student_id, req.topic, req.correct)
    return {"status": "logged"}


@app.get("/progress/{student_id}", response_model=ProgressResponse)
async def progress(student_id: str):
    weak_topics = db.get_weak_topics(student_id)
    accuracy_summary = db.get_topic_accuracy_summary(student_id)
    return ProgressResponse(weak_topics=weak_topics, accuracy_summary=accuracy_summary)


@app.get("/suggest/{student_id}", response_model=SuggestionResponse)
async def suggest(student_id: str):
    result = agent.suggest_next_topic(student_id)
    return SuggestionResponse(**result)


@app.get("/chat_history/{student_id}", response_model=ChatHistoryResponse)
async def chat_history(student_id: str):
    history = db.get_chat_history(student_id)
    return ChatHistoryResponse(history=history)


@app.delete("/chat_history/{student_id}")
async def clear_chat_history(student_id: str):
    db.clear_chat_history(student_id)
    return {"status": "cleared"}


@app.get("/health")
async def health():
    return {"status": "ok", "index_built": _index_built}
