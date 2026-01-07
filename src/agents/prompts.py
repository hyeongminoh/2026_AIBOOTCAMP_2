from langchain.prompts import PromptTemplate

planner_prompt = PromptTemplate.from_template(
    """
당신은 Query Planner 에이전트입니다. 사용자의 질문을 보고 아래 3가지 질문 유형을 판단하고, 필요한 분석 단계를 제안하세요.
- 전략 질문: SK/SKAX 등 기업 전략, 사업 방향 관련
- 트렌드 질문: 글로벌 기술 동향, 특정 기술 비교
- 정합성 질문: 기업 전략과 글로벌 트렌드의 정렬 여부

출력은 JSON으로 반환:
{{
  "intent": "strategy|trend|alignment",
  "requires_sk": true|false,
  "requires_global": true|false,
  "requires_alignment": true|false,
  "reason": "한줄 근거"
}}
질문: {question}
"""
)

alignment_prompt = PromptTemplate.from_template(
    """
당신은 Alignment Analyzer입니다. 아래의 두 정보 묶음을 비교하여 정합성을 평가하세요.
- SK 관련 정보: {sk_context}
- 글로벌 기술 트렌드: {global_context}

1) 정합성 요약 (2줄 이내)
2) 기회 요인 2가지
3) 리스크 2가지
4) 실행 권고 3가지 (우선순위 순)
형식은 한국어 불릿으로 간결하게 작성하세요.
"""
)

formatter_prompt = PromptTemplate.from_template(
    """
당신은 Answer Formatter입니다. 아래 정보를 받아 최종 응답을 구조화하세요.

질문: {question}
플랜: {plan}
SK 근거: {sk_sources}
글로벌 근거: {global_sources}
정합성 분석: {alignment}

출력 형식:
- 한줄 핵심 요약
- 상세 답변 (3~5줄)
- 근거 기사: 소스명 - 날짜 - 제목
- 다음 액션 3가지
한국어로 응답하세요.
"""
)

