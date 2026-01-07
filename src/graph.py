from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, TypedDict

from langchain.docstore.document import Document
from langchain.schema import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.prompts import alignment_prompt, formatter_prompt, planner_prompt
from src.config import get_settings
from src.rag.retrievers import get_retriever


class AgentState(TypedDict, total=False):
    question: str
    plan: Dict[str, Any]
    sk_docs: List[Document]
    global_docs: List[Document]
    alignment: str
    answer: str


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
    llm = _llm()
    chain = planner_prompt | llm | StrOutputParser()
    raw = chain.invoke({"question": state["question"]})
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "intent": "alignment",
            "requires_sk": True,
            "requires_global": True,
            "requires_alignment": True,
            "reason": "fallback parsing 실패",
        }
    return {**state, "plan": parsed}


def sk_rag_node(state: AgentState) -> AgentState:
    plan = state.get("plan", {})
    if not plan or not plan.get("requires_sk", True):
        return state

    retriever = get_retriever()
    docs = retriever.get_relevant_documents(state["question"], where={"source_type": "sk"})
    return {**state, "sk_docs": docs}


def global_rag_node(state: AgentState) -> AgentState:
    plan = state.get("plan", {})
    if not plan or not plan.get("requires_global", True):
        return state

    retriever = get_retriever()
    docs = retriever.get_relevant_documents(state["question"], where={"source_type": "global"})
    return {**state, "global_docs": docs}


def alignment_node(state: AgentState) -> AgentState:
    plan = state.get("plan", {})
    if not plan.get("requires_alignment", True):
        return {**state, "alignment": "정합성 분석이 필요하지 않음"}

    sk_context = "\n".join([d.page_content for d in state.get("sk_docs", [])][:3])
    global_context = "\n".join([d.page_content for d in state.get("global_docs", [])][:3])
    llm = _llm()
    chain = alignment_prompt | llm | StrOutputParser()
    analysis = chain.invoke({"sk_context": sk_context, "global_context": global_context})
    return {**state, "alignment": analysis}


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
    formatted = chain.invoke(
        {
            "question": state["question"],
            "plan": state.get("plan", {}),
            "sk_sources": sk_sources,
            "global_sources": global_sources,
            "alignment": state.get("alignment", ""),
        }
    )
    return {**state, "answer": formatted}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("sk_rag", sk_rag_node)
    graph.add_node("global_rag", global_rag_node)
    graph.add_node("alignment", alignment_node)
    graph.add_node("format", format_node)

    graph.set_entry_point("plan")
    graph.add_edge("plan", "sk_rag")
    graph.add_edge("sk_rag", "global_rag")
    graph.add_edge("global_rag", "alignment")
    graph.add_edge("alignment", "format")
    graph.add_edge("format", END)
    return graph.compile()


def run_agent(question: str) -> str:
    workflow = build_graph()
    result: AgentState = workflow.invoke({"question": question})
    return result.get("answer", "응답 생성에 실패했습니다.")

