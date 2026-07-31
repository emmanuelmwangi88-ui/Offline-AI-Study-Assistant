"""
models.py
---------
Pydantic schemas for FastAPI request/response bodies.
Keeping these separate from main.py keeps route definitions clean.
"""

from pydantic import BaseModel
from typing import List, Optional


class AskRequest(BaseModel):
    question: str
    student_id: Optional[str] = "student_1"


class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    model_used: Optional[str] = None
    grounded: bool = True  # False when answered without ingested notes


class ChatTurn(BaseModel):
    question: str
    answer: str
    sources: List[str]
    model_used: Optional[str] = None
    grounded: bool = True
    timestamp: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    history: List[ChatTurn]


class QuizRequest(BaseModel):
    topic: Optional[str] = None       # single topic (backward compatible)
    topics: Optional[List[str]] = None  # multiple topics for a mixed-review quiz
    num_questions: int = 3
    difficulty: str = "medium"        # "easy", "medium", or "hard"

    def resolved_topics(self) -> List[str]:
        if self.topics:
            return self.topics
        if self.topic:
            return [self.topic]
        return ["general"]


class QuizQuestion(BaseModel):
    question: str
    expected_answer: str


class QuizResponse(BaseModel):
    questions: List[QuizQuestion]
    grounded: bool = True  # False when generated without ingested notes


class GradeRequest(BaseModel):
    user_answer: str
    expected_answer: str


class GradeResponse(BaseModel):
    correct: bool
    similarity: float


class AttemptRequest(BaseModel):
    student_id: str
    topic: str
    correct: bool


class ProgressResponse(BaseModel):
    weak_topics: List[str]
    accuracy_summary: dict


class SuggestionResponse(BaseModel):
    suggested_topic: Optional[str]
    reason: str


class SessionSummaryResponse(BaseModel):
    summary: str
    topics_covered: List[str]


class NotesListResponse(BaseModel):
    sources: List[str]
