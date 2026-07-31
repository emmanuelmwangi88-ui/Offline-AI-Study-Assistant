"""
rag.py
------
Core RAG pipeline: load documents -> chunk -> embed -> store index ->
retrieve relevant chunks -> generate an answer using a local Ollama model.

This module is imported both by the FastAPI backend (main.py) and by the
Jupyter notebook, so experiments in the notebook use the exact same code
that runs in production.
"""

import os
import json
import re
import hashlib
import requests
from pathlib import Path
from typing import List, Dict

import numpy as np
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "index"
DATA_INDEX_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"

OLLAMA_MODEL = "llama3.2:latest"           # general Q&A / quiz generation
MATH_MODEL = "deepseek-r1:1.5b"            # reasoning-tuned, better at step-by-step math
VISION_MODEL = "moondream:1.8b"            # extracts text/describes uploaded images
EMBEDDING_MODEL_NAME = "nomic-embed-text"  # embedding model (all run via Ollama)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

CHUNK_SIZE = 400       # approx tokens (we use words as a simple proxy)
CHUNK_OVERLAP = 50

DEFAULT_NUM_PREDICT = 350   # caps response length — most study answers don't need more,
                            # and generation time scales roughly with output length
OLLAMA_KEEP_ALIVE = "30m"  # keeps a model loaded in memory between calls, so switching
                            # between llama3.2 / deepseek-r1 / moondream doesn't reload
                            # from disk each time (Ollama's default is only 5m)


_EMBED_CACHE_PATH = DATA_INDEX_DIR / "embed_cache.json"
_embed_cache: Dict[str, list] = {}


def _load_embed_cache():
    global _embed_cache
    if _EMBED_CACHE_PATH.exists():
        try:
            with open(_EMBED_CACHE_PATH) as f:
                _embed_cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            _embed_cache = {}


def _save_embed_cache():
    with open(_EMBED_CACHE_PATH, "w") as f:
        json.dump(_embed_cache, f)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Get embeddings for a list of texts using Ollama's local embedding model.
    Ollama's /api/embeddings endpoint takes one prompt at a time, so we loop —
    fine at this project's scale (a handful of documents, not millions).

    Results are cached on disk keyed by a hash of the text, so re-ingesting
    the same content (e.g. restarting the app, or re-running notebook cells)
    doesn't re-pay the embedding cost.
    """
    if not _embed_cache:
        _load_embed_cache()

    vectors = []
    cache_dirty = False
    for text in texts:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in _embed_cache:
            vectors.append(_embed_cache[key])
            continue

        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBEDDING_MODEL_NAME, "prompt": text, "keep_alive": OLLAMA_KEEP_ALIVE},
            timeout=60,
        )
        response.raise_for_status()
        vector = response.json()["embedding"]
        _embed_cache[key] = vector
        cache_dirty = True
        vectors.append(vector)

    if cache_dirty:
        _save_embed_cache()

    return np.array(vectors, dtype="float32")


# ---------------------------------------------------------------------------
# 1. Loading documents
# ---------------------------------------------------------------------------

def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_pdf_file(path: Path) -> str:
    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)


def load_image_file(path: Path) -> str:
    """
    Uses the local vision model (moondream) to extract text and describe
    relevant visual content from a photographed page of notes. Vision models
    like moondream aren't as precise as dedicated OCR tools on dense text,
    so this works best for legible handwriting/print — worth noting as a
    limitation in the notebook.
    """
    import base64
    image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")

    prompt = (
        "Transcribe all readable text from this image of study notes. "
        "If there are diagrams or figures, briefly describe what they show. "
        "Output only the transcription/description, no extra commentary."
    )

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": VISION_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_file(path)
    if suffix in IMAGE_EXTENSIONS:
        return load_image_file(path)
    return load_text_file(path)


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Simple word-based sliding-window chunker.
    Not perfectly token-accurate, but good enough for a first version
    and easy to reason about / tune during notebook experiments.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_text_by_paragraph(text: str, chunk_size: int = CHUNK_SIZE,
                             overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Paragraph-aware chunker: splits on blank lines first, then greedily packs
    whole paragraphs together up to chunk_size words (so related sentences
    stay together instead of being cut mid-thought by a fixed word window).
    A paragraph longer than chunk_size on its own falls back to the sliding
    window so nothing is ever dropped. This is the default chunker used by
    StudyIndex; chunk_text() is kept for notebook comparison experiments.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return chunk_text(text, chunk_size, overlap)

    chunks = []
    current_words: List[str] = []

    for para in paragraphs:
        para_words = para.split()

        if len(para_words) > chunk_size:
            # Flush whatever we were building, then window this oversized paragraph on its own
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
            chunks.extend(chunk_text(para, chunk_size, overlap))
            continue

        if len(current_words) + len(para_words) > chunk_size:
            chunks.append(" ".join(current_words))
            current_words = para_words
        else:
            current_words.extend(para_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding + Index
# ---------------------------------------------------------------------------

class StudyIndex:
    """
    Lightweight in-memory vector index using plain numpy cosine similarity
    instead of FAISS. At this project's scale (a student's own notes — tens
    to low hundreds of chunks, not millions), brute-force numpy is just as
    fast as FAISS and removes a large compiled dependency, making the whole
    app lighter to install and faster to start up.
    """

    def __init__(self):
        self.embeddings: np.ndarray = None  # shape (N, dim), L2-normalized
        self.metadata: List[Dict] = []      # one dict per row, same order

    def _embed_and_append(self, documents: List[Dict]):
        """Shared logic for build() and add(): chunk, embed, append to index."""
        all_chunks = []
        for doc in documents:
            chunks = chunk_text_by_paragraph(doc["text"])
            for c in chunks:
                all_chunks.append(c)
                self.metadata.append({
                    "source": doc.get("source", "unknown"),
                    "topic": doc.get("topic", "general"),
                    "text": c,
                })

        if not all_chunks:
            raise ValueError("No chunks produced — check input documents.")

        new_embeddings = embed_texts(all_chunks)
        norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid divide-by-zero on a degenerate embedding
        new_embeddings = new_embeddings / norms

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def build(self, documents: List[Dict]):
        """
        documents: list of {"text": str, "source": str, "topic": str}
        Builds a fresh index from scratch (clears any existing data first).
        """
        self.embeddings = None
        self.metadata = []
        self._embed_and_append(documents)

    def add(self, documents: List[Dict]):
        """
        Incrementally adds documents to an existing index without re-embedding
        anything already indexed — use this for "ingest another file" so
        ingestion time doesn't grow with how much you've already uploaded.
        Safe to call even if the index hasn't been built yet (behaves like build()).
        """
        self._embed_and_append(documents)

    def remove_source(self, source: str) -> int:
        """
        Removes all chunks belonging to a given source filename (e.g. to let
        a user delete a bad upload). Returns the number of chunks removed.
        Rebuilds the embeddings array by keeping only the surviving rows —
        cheap at this scale, and no re-embedding needed since we keep the
        existing vectors for the chunks we keep.
        """
        if self.embeddings is None:
            return 0

        keep_mask = [m["source"] != source for m in self.metadata]
        removed = len(keep_mask) - sum(keep_mask)
        if removed == 0:
            return 0

        self.metadata = [m for m, keep in zip(self.metadata, keep_mask) if keep]
        if self.metadata:
            self.embeddings = self.embeddings[keep_mask]
        else:
            self.embeddings = None
        return removed

    def list_sources(self) -> List[str]:
        """Unique source filenames currently in the index, for a 'manage notes' UI."""
        seen = []
        for m in self.metadata:
            if m["source"] not in seen:
                seen.append(m["source"])
        return seen

    def search(self, query: str, top_k: int = 3, topic: str = None) -> List[Dict]:
        if self.embeddings is None or len(self.metadata) == 0:
            raise RuntimeError("Index not built yet. Call build() first.")

        q_emb = embed_texts([query])[0]
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        # Cosine similarity via dot product (both sides already L2-normalized)
        scores = self.embeddings @ q_emb

        # When a topic is given, over-fetch and filter, falling back to the
        # unfiltered results if nothing matches that topic — cheap and avoids
        # a separate per-topic index structure at this project's scale.
        fetch_k = min((top_k * 4 if topic else top_k), len(self.metadata))
        top_idxs = np.argsort(-scores)[:fetch_k]

        results = []
        for idx in top_idxs:
            entry = dict(self.metadata[idx])
            entry["score"] = float(scores[idx])
            results.append(entry)

        if topic:
            filtered = [r for r in results if r.get("topic") == topic]
            if filtered:
                return filtered[:top_k]
            return results[:top_k]

        return results[:top_k]

    def save(self, name: str = "study_index"):
        if self.embeddings is not None:
            np.save(DATA_INDEX_DIR / f"{name}_embeddings.npy", self.embeddings)
        with open(DATA_INDEX_DIR / f"{name}_meta.json", "w") as f:
            json.dump(self.metadata, f)

    def load(self, name: str = "study_index"):
        embeddings_path = DATA_INDEX_DIR / f"{name}_embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(str(embeddings_path))
        self.embeddings = np.load(embeddings_path)
        with open(DATA_INDEX_DIR / f"{name}_meta.json") as f:
            self.metadata = json.load(f)


# ---------------------------------------------------------------------------
# 4. Generation (calls local Ollama model)
# ---------------------------------------------------------------------------

def format_chat_history(history: List[Dict]) -> str:
    """
    Format recent Q&A turns for inclusion in a prompt so the model has
    conversational context (e.g. "it", "that", follow-up questions).
    Expects a list of {"question": str, "answer": str} dicts, oldest first.
    """
    if not history:
        return ""
    lines = []
    for turn in history:
        lines.append(f"Student: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)


def build_prompt(question: str, context_chunks: List[Dict], chat_history: str = "") -> str:
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    history_section = f"\nRecent conversation:\n{chat_history}\n" if chat_history else ""
    prompt = f"""You are a study assistant helping a student understand their own notes.
Answer the question using ONLY the context provided below. If the answer
isn't in the context, say you don't have enough information rather than guessing.
Use the recent conversation (if any) to understand follow-up questions like
"what about..." or "explain that more", but still ground the actual answer in
the context below. When you use a specific fact, mention which source it
came from in brackets, like [Source: notes.txt], so the student can verify it.
{history_section}
Context:
{context_text}

Question: {question}

Answer:"""
    return prompt


def build_attachment_prompt(question: str, attachment_text: str,
                             context_chunks: List[Dict], chat_history: str = "") -> str:
    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks
    )
    history_section = f"\nRecent conversation:\n{chat_history}\n" if chat_history else ""
    prompt = f"""You are a study assistant. The student just attached a file/image to
their question. Answer using the attached content below as the primary
source, and the student's other notes as secondary context if relevant.
{history_section}
Attached content:
{attachment_text}

Other notes context (if relevant):
{context_text if context_text else "(none)"}

Question: {question}

Answer:"""
    return prompt


def call_ollama(prompt: str, model: str = OLLAMA_MODEL, num_predict: int = DEFAULT_NUM_PREDICT) -> str:
    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": {"num_predict": num_predict},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


MATH_KEYWORDS = [
    "solve", "calculate", "equation", "derivative", "integral", "algebra",
    "geometry", "theorem", "simplify", "factor", "compute", "formula",
    "divide", "subtract", "multiply", "sum of", "product of", "quotient",
    "percentage", "percent of", "square root", "how many times does",
]

# Only +, *, ^, = trigger on symbols alone — these are rare in ordinary
# English sentences. Deliberately excluding "-" and "/": they collide with
# dates ("9/11"), scores ("8/10"), fractions in recipes ("1/2 cup"), and
# ranges ("2-3 friends"), which caused false positives in testing.
MATH_SYMBOL_PATTERN = re.compile(r"[\dA-Za-z]\s*[+*^=]\s*[\dA-Za-z]")


def is_math_question(question: str) -> bool:
    """
    Simple heuristic: flag as math if it contains common math keywords/phrases,
    or has a symbolic pattern (digit/variable next to +, *, ^, or =). Word-based
    subtraction/division ("divide 10 by 2") is caught by keywords instead of
    symbols, since "-" and "/" alone are too ambiguous with dates, scores, and
    fractions in everyday phrasing (tested against "9/11", "8/10", "2-3 friends",
    "1/2 cup of sugar" — none should trigger, and none do with this version).
    Not perfect — a genuinely ambiguous case (e.g. "how many electrons does
    carbon have") will still go to the general model, an acceptable tradeoff
    for how cheap and transparent this check is.
    """
    lower = question.lower()
    if any(kw in lower for kw in MATH_KEYWORDS):
        return True
    if MATH_SYMBOL_PATTERN.search(question):
        return True
    return False


def build_fallback_prompt(question: str, chat_history: str = "") -> str:
    history_section = f"\nRecent conversation:\n{chat_history}\n" if chat_history else ""
    return f"""You are a study assistant. No notes have been uploaded yet, so
answer from general knowledge. Use the recent conversation (if any) to
understand follow-up questions. Keep the answer concise and clearly say if
you're not fully sure, rather than guessing confidently.
{history_section}
Question: {question}

Answer:"""


def answer_general(question: str, chat_history: str = "") -> Dict:
    """
    Fallback path used when no notes have been ingested. This is NOT grounded
    RAG — the model answers from its own general knowledge instead of the
    student's material. Callers should surface this distinction to the user
    (see main.py's `grounded` flag) rather than presenting it the same way
    as a grounded answer.
    """
    prompt = build_fallback_prompt(question, chat_history)
    model = MATH_MODEL if is_math_question(question) else OLLAMA_MODEL
    answer = call_ollama(prompt, model=model)
    return {
        "answer": answer,
        "sources": [],
        "context_used": [],
        "model_used": model,
    }


def answer_question(index: StudyIndex, question: str, top_k: int = 3,
                     chat_history: str = "") -> Dict:
    context_chunks = index.search(question, top_k=top_k)
    prompt = build_prompt(question, context_chunks, chat_history)

    model = MATH_MODEL if is_math_question(question) else OLLAMA_MODEL
    answer = call_ollama(prompt, model=model)

    return {
        "answer": answer,
        "sources": [c["source"] for c in context_chunks],
        "context_used": context_chunks,
        "model_used": model,
    }


def answer_with_attachment(index: StudyIndex, question: str, attachment_text: str,
                            top_k: int = 3, chat_history: str = "") -> Dict:
    """
    Answers a question about a one-off attached file/image without
    permanently ingesting it into the index. Also pulls in relevant chunks
    from any already-ingested notes as secondary context.
    """
    context_chunks = []
    if index.embeddings is not None:
        context_chunks = index.search(question, top_k=top_k)

    prompt = build_attachment_prompt(question, attachment_text, context_chunks, chat_history)
    model = MATH_MODEL if is_math_question(question) else OLLAMA_MODEL
    answer = call_ollama(prompt, model=model)

    return {
        "answer": answer,
        "sources": [c["source"] for c in context_chunks],
        "context_used": context_chunks,
        "model_used": model,
    }


# ---------------------------------------------------------------------------
# Quick manual test (run this file directly to sanity-check the pipeline)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_docs = [
        {
            "text": (
                "Photosynthesis is the process by which green plants use sunlight "
                "to synthesize food from carbon dioxide and water. It occurs mainly "
                "in the chloroplasts, using chlorophyll to capture light energy. "
                "The process produces glucose and releases oxygen as a byproduct."
            ),
            "source": "biology_notes.txt",
            "topic": "photosynthesis",
        }
    ]

    idx = StudyIndex()
    idx.build(sample_docs)

    result = answer_question(idx, "What does photosynthesis produce?")
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])
    print("Model used:", result["model_used"])
