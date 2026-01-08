from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.graph import run_agent_stream
from src.tools import search_global_it, search_sk_news


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    agents: List[str]
    tools: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str


class SearchResponse(BaseModel):
    results: list


app = FastAPI(title="SK Tech Strategy Alignment API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=app.version)


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(..., description="검색 질의"),
    source: Literal["sk", "it", "all"] = Query("all", description="검색 대상"),
    k: int = Query(3, ge=1, le=10, description="Top-k"),
) -> SearchResponse:
    results = []
    if source in {"sk", "all"}:
        results.extend(search_sk_news.invoke({"query": q, "k": k}))
    if source in {"it", "all"}:
        results.extend(search_global_it.invoke({"query": q, "k": k}))
    return SearchResponse(results=results)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")
    agents: List[str] = []
    tools: List[str] = []
    answer = "응답 생성에 실패했습니다."

    for event in run_agent_stream(req.question):
        if event["type"] == "status" and event["status"] == "start":
            if event["node"] not in agents:
                agents.append(event["node"])
        elif event["type"] == "log":
            tools.append(event["message"])
        elif event["type"] == "answer":
            answer = event.get("answer", answer)

    return AskResponse(answer=answer, agents=agents, tools=tools)

