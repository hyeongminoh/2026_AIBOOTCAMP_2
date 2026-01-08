from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from src.rag.retrievers import get_faiss_index


def _apply_filters(filters: Optional[Dict[str, Any]], source_type: str) -> Dict[str, Any]:
    """Build Chroma filter dict from generic filters."""
    clauses: List[Dict[str, Any]] = [{"source_type": source_type}]
    if filters:
        tags = filters.get("tags")
        if tags:
            clauses.append({"tags": {"$contains": ",".join(tags)}})
        # Chroma 메타데이터 타입 상 date 비교($gte)에 문자열을 쓰면 오류가 나므로
        # 현재는 날짜 필터를 적용하지 않고, 향후 숫자(yyyymmdd) 필드를 추가한 뒤 사용하도록 남겨둔다.
        # days = filters.get("days")
        # if days:
        #     try:
        #         cutoff = datetime.utcnow() - timedelta(days=int(days))
        #         clauses.append({"date_num": {"$gte": int(cutoff.strftime("%Y%m%d"))}})
        #     except Exception:
        #         pass
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _search(query: str, source_type: str, filters: Optional[Dict[str, Any]] = None, k: int = 3):
    index = get_faiss_index(source_type)
    docs = index.similarity_search(query, k=k)
    results = []
    for d in docs:
        results.append(
            {
                "content": d.page_content,
                "metadata": d.metadata,
                "score": d.metadata.get("score") or None,
            }
        )
    return results


@tool
def search_sk_news(query: str, filters: Optional[Dict[str, Any]] = None, k: int = 3) -> List[Dict[str, Any]]:
    """SK/SKAX 관련 뉴스/보도자료를 RAG 검색합니다."""
    return _search(query, "sk", filters, k)


@tool
def search_global_it(query: str, filters: Optional[Dict[str, Any]] = None, k: int = 3) -> List[Dict[str, Any]]:
    """글로벌 IT 트렌드/기사 데이터를 RAG 검색합니다."""
    return _search(query, "global", filters, k)


@tool
def classify_query(user_query: str) -> Dict[str, Any]:
    """질문을 분류하여 전략/트렌드/정합성 여부와 필요한 소스를 판단합니다."""
    text = user_query.lower()
    is_sk = any(x in text for x in ["sk", "sk텔레콤", "skt", "sk하이닉스", "sk ax"])
    mentions_align = any(x in text for x in ["정합", "aligned", "alignment", "비교", "맞는지", "맞춰", "align"])
    mentions_global = any(x in text for x in ["글로벌", "세계", "국제", "트렌드", "trend", "모델", "llm"])

    intent = "trend"
    if mentions_align and (is_sk or mentions_global):
        intent = "alignment"
    elif is_sk:
        intent = "strategy"

    return {
        "intent": intent,
        "requires_sk": bool(is_sk or intent in {"strategy", "alignment"}),
        "requires_global": bool(mentions_global or intent in {"trend", "alignment"}),
        "requires_alignment": bool(intent == "alignment"),
        "reason": "heuristic classification",
    }


@tool
def score_alignment(sk_summary: str, global_summary: str) -> Dict[str, Any]:
    """정합성 점수를 0~100으로 산출하고 라벨을 반환합니다."""
    if not sk_summary and not global_summary:
        return {"score": 0, "label": "Lagging", "reasons": ["자료 부재로 판단 불가"]}
    overlap = len(set(sk_summary.lower().split()) & set(global_summary.lower().split()))
    total = max(len(sk_summary.split()) + len(global_summary.split()), 1)
    ratio = overlap / total
    score = int(min(100, max(0, ratio * 200)))
    if score >= 67:
        label = "Leading"
    elif score >= 40:
        label = "Aligned"
    else:
        label = "Lagging"
    reasons = [
        f"중복 단어 비율 {ratio:.2f}",
        f"SK 길이 {len(sk_summary.split())}, Global 길이 {len(global_summary.split())}",
    ]
    return {"score": score, "label": label, "reasons": reasons}
