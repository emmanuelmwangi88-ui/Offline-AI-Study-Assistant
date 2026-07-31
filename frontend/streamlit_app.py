"""
streamlit_app.py
----------------
Streamlit frontend for the offline study assistant.
Talks to the FastAPI backend (see backend/app/main.py) over HTTP.

Run with:
    streamlit run frontend/streamlit_app.py

Make sure the backend is running first:
    uvicorn app.main:app --reload   (from inside backend/)
"""

import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"  # change if backend runs elsewhere

st.set_page_config(page_title="Offline Study Assistant", page_icon="📚", layout="wide")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "student_id" not in st.session_state:
    st.session_state.student_id = "student_1"  # simple single-user demo for now

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "quiz_topics_label" not in st.session_state:
    st.session_state.quiz_topics_label = []

if "chat_history" not in st.session_state:
    try:
        resp = requests.get(f"{BACKEND_URL}/chat_history/{st.session_state.student_id}")
        resp.raise_for_status()
        st.session_state.chat_history = resp.json().get("history", [])
    except requests.exceptions.RequestException:
        st.session_state.chat_history = []  # backend not up yet — start empty, non-fatal

# ---------------------------------------------------------------------------
# Sidebar: upload notes, manage notes, progress, session summary, export
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("📥 Upload Notes")
    st.caption("Upload material to quiz yourself and ask questions on — text, PDF, or a photo of your notes.")
    uploaded_file = st.file_uploader(
        "Upload a PDF, text file, or photo of your notes",
        type=["pdf", "txt", "png", "jpg", "jpeg", "webp"],
    )
    topic_label = st.text_input("Topic label for this file", value="general")

    if uploaded_file is not None and uploaded_file.type.startswith("image"):
        st.caption("Image detected — text will be extracted using the local vision model (may take longer).")

    if uploaded_file is not None and st.button("Ingest notes"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        data = {"topic": topic_label}
        with st.spinner("Processing notes..."):
            try:
                resp = requests.post(f"{BACKEND_URL}/ingest", files=files, data=data)
                resp.raise_for_status()
                result = resp.json()
                st.success("Notes ingested successfully.")
                if result.get("warning"):
                    st.warning(result["warning"])
            except requests.exceptions.RequestException as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.header("🗂️ Manage Notes")
    st.caption("See what's been ingested, or remove something you don't want the assistant using anymore.")
    if st.button("Refresh notes list"):
        try:
            resp = requests.get(f"{BACKEND_URL}/notes")
            resp.raise_for_status()
            st.session_state.notes_list = resp.json().get("sources", [])
        except requests.exceptions.RequestException as e:
            st.error(f"Could not fetch notes: {e}")

    for src in st.session_state.get("notes_list", []):
        cols = st.columns([3, 1])
        cols[0].write(f"📄 {src}")
        if cols[1].button("🗑️", key=f"delete_{src}"):
            try:
                resp = requests.delete(f"{BACKEND_URL}/notes/{src}")
                resp.raise_for_status()
                st.session_state.notes_list.remove(src)
                st.rerun()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not delete {src}: {e}")

    if st.session_state.get("notes_list") and st.button("Clear all notes", type="secondary"):
        try:
            requests.delete(f"{BACKEND_URL}/notes")
            st.session_state.notes_list = []
            st.rerun()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not clear notes: {e}")

    st.divider()
    st.header("📊 Progress")
    st.caption("Tracks your quiz accuracy per topic and flags what to review next.")
    if st.button("Refresh progress"):
        try:
            resp = requests.get(f"{BACKEND_URL}/progress/{st.session_state.student_id}")
            resp.raise_for_status()
            progress = resp.json()
            if progress.get("weak_topics"):
                st.write("**Weak topics:**")
                for t in progress["weak_topics"]:
                    st.write(f"- {t}")
            else:
                st.write("No data yet — take a quiz first.")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not fetch progress: {e}")

    st.divider()
    st.header("📝 Session Summary")
    st.caption("A quick recap of what you've covered and how you're doing.")
    if st.button("Generate summary"):
        with st.spinner("Summarizing..."):
            try:
                resp = requests.get(f"{BACKEND_URL}/session_summary/{st.session_state.student_id}")
                resp.raise_for_status()
                st.session_state.session_summary = resp.json()
            except requests.exceptions.RequestException as e:
                st.error(f"Could not generate summary: {e}")

    if st.session_state.get("session_summary"):
        st.write(st.session_state.session_summary["summary"])
        topics = st.session_state.session_summary.get("topics_covered")
        if topics:
            st.caption(f"Topics: {', '.join(topics)}")

    st.divider()
    st.header("⬇️ Export")
    if st.button("Prepare study summary"):
        try:
            resp = requests.get(f"{BACKEND_URL}/export/{st.session_state.student_id}")
            resp.raise_for_status()
            st.session_state.export_content = resp.text
        except requests.exceptions.RequestException as e:
            st.error(f"Could not prepare export: {e}")

    if st.session_state.get("export_content"):
        st.download_button(
            "Download study summary (.md)",
            data=st.session_state.export_content,
            file_name=f"{st.session_state.student_id}_study_summary.md",
            mime="text/markdown",
        )

# ---------------------------------------------------------------------------
# Main: chat / Q&A
# ---------------------------------------------------------------------------

st.title("📚 Offline Study Assistant")
st.caption("Ask questions about your notes, or take a quiz to test yourself.")

tab_ask, tab_quiz = st.tabs(["Ask a question", "Quiz me"])

with tab_ask:
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            caption_bits = []
            if turn.get("sources"):
                caption_bits.append(f"Sources: {', '.join(set(turn['sources']))}")
            if turn.get("model_used"):
                caption_bits.append(f"Model: {turn['model_used']}")
            if caption_bits:
                st.caption(" · ".join(caption_bits))

    prompt_data = st.chat_input(
        "What do you want to know? (attach a 📎 file/image to ask about it)",
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "webp", "pdf", "txt"],
    )

    if prompt_data:
        question = prompt_data.text or ""
        attached_file = prompt_data.files[0] if prompt_data.files else None

        with st.chat_message("user"):
            if attached_file is not None:
                if attached_file.type and attached_file.type.startswith("image"):
                    st.image(attached_file.getvalue(), width=220)
                else:
                    st.caption(f"📎 {attached_file.name}")
            if question:
                st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..." if not attached_file else "Reading attachment..."):
                try:
                    if attached_file is not None:
                        files = {"file": (attached_file.name, attached_file.getvalue())}
                        data = {
                            "question": question or "Describe this and summarize the key points.",
                            "student_id": st.session_state.student_id,
                        }
                        resp = requests.post(f"{BACKEND_URL}/ask_with_file", data=data, files=files)
                    else:
                        resp = requests.post(
                            f"{BACKEND_URL}/ask",
                            json={"question": question, "student_id": st.session_state.student_id},
                        )
                    resp.raise_for_status()
                    result = resp.json()
                    st.write(result["answer"])
                    caption_bits = []
                    if result.get("sources"):
                        caption_bits.append(f"Sources: {', '.join(set(result['sources']))}")
                    if result.get("model_used"):
                        caption_bits.append(f"Model: {result['model_used']}")
                    if caption_bits:
                        st.caption(" · ".join(caption_bits))

                    st.session_state.chat_history.append({
                        "question": question if question else f"[Attached: {attached_file.name}]",
                        "answer": result["answer"],
                        "sources": result.get("sources", []),
                        "model_used": result.get("model_used"),
                        "grounded": result.get("grounded", True),
                    })
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")

    if st.session_state.chat_history and st.button("Clear chat history"):
        try:
            requests.delete(f"{BACKEND_URL}/chat_history/{st.session_state.student_id}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not clear history on server: {e}")
        st.session_state.chat_history = []
        st.rerun()

with tab_quiz:
    topics_input = st.text_input(
        "Topic(s) to be quizzed on — separate with commas for a mixed-review quiz",
        value=topic_label,
    )
    topics_list = [t.strip() for t in topics_input.split(",") if t.strip()] or ["general"]

    col1, col2 = st.columns(2)
    with col1:
        num_questions = st.slider("Number of questions", 1, 10, 3)
    with col2:
        difficulty = st.select_slider("Difficulty", options=["easy", "medium", "hard"], value="medium")

    flashcard_mode = st.toggle("Flashcard mode (flip to reveal instead of typing an answer)")

    if st.button("Generate quiz"):
        with st.spinner("Generating quiz..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/quiz",
                    json={"topics": topics_list, "num_questions": num_questions, "difficulty": difficulty},
                )
                resp.raise_for_status()
                st.session_state.quiz = resp.json()["questions"]
                st.session_state.quiz_topics_label = topics_list
            except requests.exceptions.RequestException as e:
                st.error(f"Could not generate quiz: {e}")

    def _log_attempt(correct: bool):
        for t in st.session_state.quiz_topics_label:
            try:
                requests.post(
                    f"{BACKEND_URL}/attempt",
                    json={"student_id": st.session_state.student_id, "topic": t, "correct": correct},
                )
            except requests.exceptions.RequestException:
                pass  # non-fatal — quiz UX shouldn't break if logging fails

    if st.session_state.quiz:
        st.write("### Quiz")

        if flashcard_mode:
            for i, q in enumerate(st.session_state.quiz):
                with st.container(border=True):
                    st.write(f"**{i + 1}. {q['question']}**")
                    show_key = f"flip_{i}"
                    if st.button("Show answer", key=f"show_{i}"):
                        st.session_state[show_key] = True

                    if st.session_state.get(show_key):
                        st.info(q.get("expected_answer", "(no answer provided)"))
                        c1, c2 = st.columns(2)
                        if c1.button("✅ I got it right", key=f"right_{i}"):
                            _log_attempt(True)
                            st.success("Logged.")
                        if c2.button("❌ I got it wrong", key=f"wrong_{i}"):
                            _log_attempt(False)
                            st.info("Logged — this'll show up under weak topics.")
        else:
            for i, q in enumerate(st.session_state.quiz):
                st.write(f"**{i + 1}. {q['question']}**")
                answer = st.text_input(f"Your answer #{i + 1}", key=f"quiz_answer_{i}")

                if st.button(f"Submit answer #{i + 1}", key=f"submit_{i}"):
                    try:
                        resp = requests.post(
                            f"{BACKEND_URL}/grade",
                            json={"user_answer": answer, "expected_answer": q.get("expected_answer", "")},
                        )
                        resp.raise_for_status()
                        grade = resp.json()
                        _log_attempt(grade["correct"])
                        if grade["correct"]:
                            st.success("Correct! ✅")
                        else:
                            st.error(f"Not quite — expected: {q.get('expected_answer', '(none)')}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Could not grade answer: {e}")
