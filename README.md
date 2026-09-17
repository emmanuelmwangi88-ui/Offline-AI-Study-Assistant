# Offline Study Assistant (SDG 4: Quality Education)

An AI study assistant that runs entirely offline using local models via Ollama.
Students upload their own notes (text, PDF, or photos), ask grounded questions
(RAG), get quizzed on the material with fuzzy grading, get math questions
routed to a reasoning-tuned model, and receive spaced-repetition-style
suggestions on what to review next (agentic layer) — with conversation memory
and chat history persisted across sessions.

## Project structure

```
study-assistant/
├── backend/
│   ├── app/
│   │   ├── main.py      # FastAPI routes
│   │   ├── rag.py       # chunking, embedding, retrieval, generation, math/image routing
│   │   ├── agent.py     # quiz/grading/tracking/summary tools + routing
│   │   ├── db.py        # SQLite persistence (attempts + chat history + priority scoring)
│   │   └── models.py    # request/response schemas
│   └── data/
│       ├── raw/         # uploaded notes land here
│       └── index/       # saved vector index (numpy) + embedding cache
├── frontend/
│   └── streamlit_app.py # upload / chat / quiz / manage notes / export UI
├── notebook/
│   └── study_assistant_dev.ipynb   # required notebook deliverable
├── requirements.txt                 # single file — everything needed to run the app and notebook
```

## Features

**Core RAG + chat**
- RAG Q&A over your own uploaded notes, grounded in retrieved chunks only,
  with inline `[Source: filename]` citations in answers
- Conversation memory — recent Q&A turns are fed back into each new prompt,
  so follow-up questions like "what about..." actually have context
- Chat history persisted in SQLite, reloads automatically when you reopen the app
- Attach a file/image directly in the chat box (📎 icon) to ask a one-off
  question about it, without permanently adding it to your notes
- Works even with zero notes ingested — falls back to a general-knowledge
  answer instead of erroring

**Quizzing**
- Grounded quiz questions, with one automatic retry if the model's JSON
  response comes back malformed
- **Multi-topic mixed-review quizzes** — comma-separate topics for a quiz
  spanning more than one subject
- **Difficulty levels** (easy/medium/hard) — adjusts the prompt's instruction
  to the model
- **Fuzzy answer grading** — no more exact-string-match; paraphrases, typos,
  and case differences are graded correctly via similarity scoring
- **Flashcard mode** — flip to reveal the answer and self-grade, instead of typing

**Progress tracking**
- Spaced-repetition-style suggestions — combines accuracy AND how long it's
  been since you last reviewed a topic, not just a flat "weakest topic" pick
- **Study session summary** — a short model-generated recap of what you
  covered and how you did (genuinely agentic: synthesis, not retrieval)
- **Export a study summary** as a downloadable markdown file (topic accuracy
  table + recent questions)

**Notes management**
- **View and delete individual ingested notes**, or clear everything at once
- Image upload (sidebar) — a local vision model transcribes photographed
  notes, with a warning if the transcription looks suspiciously short
  (likely a blurry/unreadable photo)

**Routing**
- Math questions automatically route to a reasoning-tuned model instead of
  the general one

**Fully offline** — everything runs through local Ollama models, no cloud API calls

## Performance & size improvements

- **No FAISS** — vector search now uses plain numpy cosine similarity
  instead of the FAISS library. At this project's scale (a student's own
  notes — tens to low hundreds of chunks), brute-force numpy is just as
  fast and removes a large compiled dependency, making install lighter and faster.
- **Incremental indexing** — uploading a new file no longer re-embeds every
  previously ingested document; only the new content gets embedded
- **Embedding cache on disk** — re-ingesting identical content (restarts,
  re-running notebook cells) reuses cached embeddings instead of re-calling Ollama
- **Index persists across restarts** — the backend reloads your ingested
  notes from disk on startup instead of starting empty every time
- **Paragraph-aware chunking** — chunks respect paragraph boundaries instead
  of a blind fixed-word window (falls back to a word-window for any single
  oversized paragraph)
- **`num_predict` caps** on generation — responses don't ramble past what a
  study answer needs, which directly reduces response time
- **`keep_alive: 30m`** on every Ollama call — keeps models warm in memory
  between requests instead of reloading from disk each time you switch
  between the general, math, and vision models (Ollama's default is only 5m)

## Ollama models used (all already pulled per your setup — nothing new to download)

| Purpose | Model |
|---|---|
| General Q&A / quiz generation / session summaries | `llama3.2:latest` |
| Math question reasoning | `deepseek-r1:1.5b` |
| Image transcription/description | `moondream:1.8b` |
| Embeddings for RAG retrieval | `nomic-embed-text` |

## Things to do on your own

1. **Create a virtual environment and install dependencies:**
   ```
   python -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Make sure Ollama is running** (`ollama serve`, or it may already be running
   in the background) — everything here, including embeddings, runs through
   your local Ollama install, so no internet is needed once dependencies are installed.
3. **Test the RAG core standalone** before wiring anything else up:
   ```
   cd backend
   python app/rag.py
   ```
4. **Run the backend:**
   ```
   cd backend
   uvicorn app.main:app --reload
   ```
   Visit http://localhost:8000/health to confirm it's up.
5. **Run the frontend** (in a separate terminal):
   ```
   cd frontend
   streamlit run streamlit_app.py
   ```
6. **Try the full flow:** upload a note → ask a question → try a math
   question → generate a mixed-topic quiz at "hard" difficulty → try
   flashcard mode → check progress → generate a session summary → download
   the export → delete a note from "Manage Notes" → confirm it's gone.
7. **Replace the sample notes** in the notebook with your own real study
   material, then re-run it top to bottom.
8. **Run the model evaluation section** with Ollama running, and fill in
   your own observations — this needs your judgment, not just code.
9. **Fill in the Google Form** using the project details and what you
   actually observed running it.

