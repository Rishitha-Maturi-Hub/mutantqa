"""FastAPI wrapper: POST /chat {"question": ...} -> {"answer": ..., "sources": [...]}.

    uvicorn demo_app.main:app --port 8000
    MUTANT=m3 uvicorn demo_app.main:app --port 8000     # run a broken variant
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .rag import MUTANTS, answer, current_settings

app = FastAPI(title="HTTPX docs bot (mutantqa demo)")


class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    return answer(req.question)


@app.get("/health")
def health() -> dict:
    s = current_settings()
    return {"ok": True, "model": s.model, "mutants": {k: v[0] for k, v in MUTANTS.items()}}
