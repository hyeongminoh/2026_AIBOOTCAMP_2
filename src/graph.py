from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List, Optional, TypedDict

from langchain.docstore.document import Document
from langchain.schema import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.prompts import alignment_prompt, formatter_prompt, planner_prompt
from src.config import get_settings
from src.rag.retrievers import get_retriever
from src.tools import classify_query, score_alignment, search_global_it, search_sk_news


class AgentState(TypedDict, total=False):
    question: str
    plan: Dict[str, Any]
    sk_docs: List[Document]
    global_docs: List[Document]
    alignment: str
    alignment_score: Dict[str, Any]
    answer: str
    logs: List[str]
    history: List[Dict[str, str]]


def _llm() -> AzureChatOpenAI:
    settings = get_settings()
    return AzureChatOpenAI(
        azure_deployment=settings.azure.chat_deployment,
        api_key=settings.azure.api_key,
        azure_endpoint=settings.azure.endpoint,
        api_version=settings.azure.api_version,
        temperature=0.1,
    )


def plan_node(state: AgentState) -> AgentState:
    def _normalize(p: Dict[str, Any]) -> Dict[str, Any]:
        intent = p.get("intent") or "trend"
        requires_sk = p.get("requires_sk")
        requires_global = p.get("requires_global")
        requires_alignment = p.get("requires_alignment")
        if intent in {"strategy", "alignment"} and requires_sk is None:
            requires_sk = True
        if intent in {"trend", "alignment"} and requires_global is None:
            requires_global = True
        if requires_alignment is None:
            requires_alignment = intent == "alignment"
        return {
            "intent": intent,
            "requires_sk": bool(requires_sk),
            "requires_global": bool(requires_global),
            "requires_alignment": bool(requires_alignment),
            "reason": p.get("reason", "normalized"),
        }

    logs = list(state.get("logs", []))
    plan: Optional[Dict[str, Any]] = None

    # 1) LLM 기반 JSON 플래너 우선
    chain = planner_prompt | _llm() | StrOutputParser()
    raw = chain.invoke({"question": state["question"]})
    try:
        plan = _normalize(json.loads(raw))
        logs.append(
            f"📄 planner_prompt JSON → intent={plan.get('intent')} "
            f"sk={plan.get('requires_sk')} global={plan.get('requires_global')} "
            f"align={plan.get('requires_alignment')}"
        )
    except json.JSONDecodeError:
        plan = None

    # 2) 실패 시 classify_query 툴 호출
    if plan is None:
        llm = _llm().bind_tools([classify_query])
        msg = llm.invoke(state["question"])
        tool_calls = getattr(msg, "tool_calls", []) or []
        for call in tool_calls:
            if call.get("name") == "classify_query":
                plan = _normalize(
                    classify_query.invoke({"user_query": state["question"]})
                )
                logs.append(
                    f"🔧 classify_query 호출 → intent={plan.get('intent')} "
                    f"sk={plan.get('requires_sk')} global={plan.get('requires_global')} "
                    f"align={plan.get('requires_alignment')}"
                )
                break

    # 3) 최종 실패 시 안전한 기본값
    if plan is None:
        plan = {
            "intent": "trend",
            "requires_sk": False,
            "requires_global": True,
            "requires_alignment": False,
            "reason": "fallback parsing 실패",
        }
        logs.append("⚠️ planner JSON/툴 모두 실패 → 기본 플랜 사용")

    return {**state, "plan": plan, "logs": logs}


def react_refine_node(state: AgentState) -> AgentState:
    """
    ReAct 스타일로 한 번 더 검색 질의를 정제해 필요한 경우 SK/Global 검색 툴을 즉시 호출.
    """
    plan = state.get("plan", {})
    if not (plan.get("requires_sk") or plan.get("requires_global")):
        return state

    logs = list(state.get("logs", []))
    sk_docs = list(state.get("sk_docs", []))
    global_docs = list(state.get("global_docs", []))

    llm = _llm().bind_tools([search_sk_news, search_global_it])
    prompt = (
        "너는 ReAct 검색 보조 에이전트다. 질문과 플랜을 보고 필요하면 검색 툴을 호출하라.\n"
        "규칙:\n"
        "- SK 필요하면 search_sk_news(query=세부 질의, filters=None, k=2)를 1회 호출\n"
        "- 글로벌 필요하면 search_global_it(query=세부 질의, filters=None, k=2)를 1회 호출\n"
        "- 불필요한 툴 호출은 하지 않는다.\n"
        "질문: {question}\n플랜: {plan}\n"
    )
    msg = llm.invoke(prompt.format(question=state["question"], plan=plan))
    tool_calls = getattr(msg, "tool_calls", []) or []

    for call in tool_calls:
        name = call.get("name")
        args = call.get("args") or {}
        if name == "search_sk_news":
            res = search_sk_news.invoke({"query": args.get("query") or state["question"], "k": 2})
            logs.append("🔧 ReAct search_sk_news 호출")
            for r in res:
                sk_docs.append(Document(page_content=r["content"], metadata=r["metadata"]))
        elif name == "search_global_it":
            res = search_global_it.invoke({"query": args.get("query") or state["question"], "k": 2})
            logs.append("🔧 ReAct search_global_it 호출 (IT 트렌드)")
            for r in res:
                global_docs.append(Document(page_content=r["content"], metadata=r["metadata"]))

    return {**state, "logs": logs, "sk_docs": sk_docs, "global_docs": global_docs}


def sk_rag_node(state: AgentState) -> AgentState:
    plan = state.get("plan", {})
    if not plan or not plan.get("requires_sk", True):
        return state

    logs = list(state.get("logs", []))
    results = search_sk_news.invoke({"query": state["question"], "filters": {"days": 180}, "k": 3})
    logs.append(f"🔧 search_sk_news 호출 (k=3, last=180d)")
    docs = []
    for r in results:
        docs.append(
            Document(
                page_content=r["content"],
                metadata=r["metadata"],
            )
        )
    return {**state, "sk_docs": docs, "logs": logs}


def global_rag_node(state: AgentState) -> AgentState:
    plan = state.get("plan", {})
    if not plan or not plan.get("requires_global", True):
        return state

    logs = list(state.get("logs", []))
    # SK가 먼저 실행된 경우, SK 컨텍스트를 힌트로 함께 전달하여 IT 검색을 보강
    sk_hint = ""
    if state.get("sk_docs"):
        sk_titles = [d.metadata.get("title", "") for d in state["sk_docs"][:2]]
        sk_hint = " | SK 컨텍스트: " + " / ".join(filter(None, sk_titles))

    results = search_global_it.invoke(
        {"query": state["question"] + sk_hint, "filters": {"days": 180}, "k": 3}
    )
    logs.append(f"🔧 search_global_it 호출 (k=3, last=180d){' with SK hint' if sk_hint else ''}")
    docs = []
    for r in results:
        docs.append(
            Document(
                page_content=r["content"],
                metadata=r["metadata"],
            )
        )
    return {**state, "global_docs": docs, "logs": logs}


def alignment_node(state: AgentState) -> AgentState:
    plan = state.get("plan", {})
    if not plan.get("requires_alignment", True):
        return {**state, "alignment": "정합성 분석이 필요하지 않음"}

    logs = list(state.get("logs", []))
    sk_context = "\n".join([d.page_content for d in state.get("sk_docs", [])][:3])
    global_context = "\n".join([d.page_content for d in state.get("global_docs", [])][:3])
    llm = _llm()
    chain = alignment_prompt | llm | StrOutputParser()
    analysis = chain.invoke({"sk_context": sk_context, "global_context": global_context})

    score = score_alignment.invoke({"sk_summary": sk_context, "global_summary": global_context})
    logs.append(f"🔧 score_alignment 호출 → {score.get('label')} ({score.get('score')})")
    return {**state, "alignment": analysis, "alignment_score": score, "logs": logs}


def format_node(state: AgentState) -> AgentState:
    llm = _llm()
    chain = formatter_prompt | llm | StrOutputParser()
    sk_sources = [
        f"{d.metadata.get('source')} - {d.metadata.get('date')} - {d.metadata.get('title')}"
        for d in state.get("sk_docs", [])[:3]
    ]
    global_sources = [
        f"{d.metadata.get('source')} - {d.metadata.get('date')} - {d.metadata.get('title')}"
        for d in state.get("global_docs", [])[:3]
    ]
    history_items = state.get("history", []) or []
    history_text = "\n".join(
        [f"Q: {h.get('q')}\nA: {h.get('a')}" for h in history_items[-3:]]  # 최근 3개
    )
    formatted = chain.invoke(
        {
            "question": state["question"],
            "plan": state.get("plan", {}),
            "sk_sources": sk_sources,
            "global_sources": global_sources,
            "alignment": state.get("alignment", ""),
            "alignment_score": state.get("alignment_score", {}),
            "history": history_text,
        }
    )
    return {**state, "answer": formatted}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", plan_node)
    graph.add_node("react_refine", react_refine_node)
    graph.add_node("sk_rag", sk_rag_node)
    graph.add_node("global_rag", global_rag_node)
    graph.add_node("alignment_step", alignment_node)
    graph.add_node("format", format_node)

    graph.set_entry_point("planner")

    def route_from_planner(state: AgentState) -> str:
        plan = state.get("plan", {})
        if plan.get("requires_sk", True):
            return "sk_rag"
        if plan.get("requires_global", True):
            return "global_rag"
        if plan.get("requires_alignment", True):
            return "alignment_step"
        return "format"

    def route_from_sk(state: AgentState) -> str:
        plan = state.get("plan", {})
        if plan.get("requires_global", True):
            return "global_rag"
        if plan.get("requires_alignment", True):
            return "alignment_step"
        return "format"

    def route_from_global(state: AgentState) -> str:
        plan = state.get("plan", {})
        if plan.get("requires_alignment", True):
            return "alignment_step"
        return "format"

    graph.add_edge("planner", "react_refine")
    graph.add_conditional_edges(
        "react_refine",
        route_from_planner,
        {
            "sk_rag": "sk_rag",
            "global_rag": "global_rag",
            "alignment_step": "alignment_step",
            "format": "format",
            "__else__": "format",
        },
    )
    graph.add_conditional_edges(
        "sk_rag",
        route_from_sk,
        {
            "global_rag": "global_rag",
            "alignment_step": "alignment_step",
            "format": "format",
            "__else__": "format",
        },
    )
    graph.add_conditional_edges(
        "global_rag",
        route_from_global,
        {"alignment_step": "alignment_step", "format": "format", "__else__": "format"},
    )
    graph.add_edge("alignment_step", "format")
    graph.add_edge("format", END)
    return graph.compile()


def run_agent(question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    workflow = build_graph()
    result: AgentState = workflow.invoke({"question": question, "history": history or []})
    return result.get("answer", "응답 생성에 실패했습니다.")


def run_agent_stream(
    question: str, history: Optional[List[Dict[str, str]]] = None
) -> Iterator[Dict[str, Any]]:
    """
    노드 진행 상황을 순차적으로 방출하는 제너레이터.
    Streamlit 등에서 단계별 상태 표시용으로 사용.
    """
    workflow = build_graph()
    result_state: AgentState = {}
    last_log_count = 0

    for event in workflow.stream({"question": question, "history": history or []}):
        for node, state in event.items():
            if node == "__end__":
                continue
            # 실행된 노드의 start/end를 함께 알림
            yield {"type": "status", "node": node, "status": "start"}
            # 도구 로그가 있다면 추가 방출
            logs = state.get("logs", [])
            if logs:
                new_logs = logs[last_log_count:]
                last_log_count = len(logs)
                for log in new_logs:
                    yield {"type": "log", "message": log}
            yield {"type": "status", "node": node, "status": "end"}
            result_state = state

    yield {"type": "answer", "answer": result_state.get("answer", "응답 생성에 실패했습니다.")}

