"""FastAPI service for dual bio graph rag question answering."""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI

app = FastAPI(title="dual bio graph rag QA")
LOGGER = logging.getLogger(__name__)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/qa")
def qa_endpoint(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = payload.get("question")
    if not question:
        return {"error": "question is required"}
    # Placeholder: integrate retrieval + LLM pipeline
    LOGGER.info("Received question: %s", question)
    return {"question": question, "answer": "TODO: integrate with GraphRAG pipeline"}

