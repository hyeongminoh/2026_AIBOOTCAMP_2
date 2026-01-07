from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.graph import run_agent


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


app = FastAPI(title="SK Tech Strategy Alignment API", version="0.1.0")


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")
    answer = run_agent(req.question)
    return AskResponse(answer=answer)

