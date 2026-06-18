from fastapi import FastAPI
from pydantic import BaseModel
from implementation.answer import answer_question
import time

app = FastAPI(title="CMS RAG API")


class AskRequest(BaseModel):
    question: str
    history: list = []



@app.post("/ask")
def ask(request: AskRequest):
    start_time = time.perf_counter()
    answer, sources = answer_question(request.question, request.history)
    end_time = time.perf_counter()
    return {
        'answer': answer,
        'sources':sources,
        'latency': end_time - start_time
    }




@app.get("/health")
def check_health():
    return {"status": "ok"}