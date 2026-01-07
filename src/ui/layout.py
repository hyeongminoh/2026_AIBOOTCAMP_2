from __future__ import annotations

from typing import List, Tuple

import streamlit as st

from src.graph import run_agent_stream
from src.ui.state import ensure_state


ICONS = {
    "planner": "🧭 Planner",
    "sk_rag": "📰 SK RAG",
    "global_rag": "🧠 IT RAG",
    "alignment_step": "⚖️ Alignment",
    "format": "🧩 Formatter",
}

LOGO_PATH = "src/ui/SKLOGO.png"


def _inject_css():
    st.markdown(
        """
        <style>
        .answer-card {
            padding: 16px 18px;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            background: #f9fafb;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            margin-bottom: 12px;
        }
        .answer-card .answer-title {
            font-weight: 700;
            margin-bottom: 8px;
            font-size: 1.05rem;
        }
        .answer-card .answer-body {
            white-space: pre-wrap;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_state():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts: {"q": ..., "a": ...}


def render_sidebar():
    with st.sidebar:
        st.subheader("대화 기록")
        if st.session_state.history:
            for idx, item in enumerate(reversed(st.session_state.history[-10:]), 1):  # 최근 10개만 표시
                title = f"{idx}. {item['q'][:40]}{'...' if len(item['q'])>40 else ''}"
                with st.expander(title, expanded=False):
                    st.markdown(f"**Q:** {item['q']}")
                    st.markdown(f"**A:** {item['a']}")
        else:
            st.info("기록 없음")


def render_main(question: str):
    st.markdown(f"**입력한 질문:** {question}")
    answer_placeholder = st.container()
    summary_container = st.container()
    log_container = st.container()
    logs: List[Tuple[str, str]] = []
    agent_runs = []
    tool_logs: List[str] = []
    tool_seen = set()

    def append_log(msg: str, kind: str = "info"):
        logs.append((kind, msg))
        log_container.empty()
        for level, text in logs:
            if level == "success":
                log_container.success(text)
            elif level == "warning":
                log_container.warning(text)
            elif level == "error":
                log_container.error(text)
            else:
                log_container.info(text)

    try:
        for event in run_agent_stream(question, history=st.session_state.history):
            if event["type"] == "status" and event["status"] == "start":
                if event["node"] not in agent_runs:
                    agent_runs.append(event["node"])
                    append_log(f"{ICONS.get(event['node'], event['node'])} 수행", "info")
            elif event["type"] == "log":
                if event["message"] not in tool_seen:
                    tool_seen.add(event["message"])
                    tool_logs.append(event["message"])
                    append_log(event["message"], "info")
            elif event["type"] == "answer":
                # 실제 실행된 에이전트/툴 요약만 표시
                if agent_runs:
                    agent_list = ", ".join([ICONS.get(a, a) for a in agent_runs])
                    summary_container.success(f"실행한 에이전트: {agent_list}")
                if tool_logs:
                    summary_container.info("사용한 툴:\n" + "\n".join(tool_logs))
                # 답변을 카드 형태로 표시
                answer_placeholder.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="answer-title">답변</div>
                        <div class="answer-body">{event['answer']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # 히스토리 저장 (메모리)
                st.session_state.history.append({"q": question, "a": event["answer"]})
                # 질문 입력창 초기화
                st.session_state.question_input = ""
        st.success("완료")
    except Exception as e:
        append_log(f"오류 발생: {e}", "error")
        st.error(f"오류 발생: {e}")


def render_page():
    st.set_page_config(
        page_title="SK TECH & IT TREND AGENT",
        layout="wide",
        page_icon=LOGO_PATH,
    )
    header_cols = st.columns([1, 5])
    with header_cols[0]:
        st.image(LOGO_PATH, width=80)
    with header_cols[1]:
        st.title("SK TECH & IT TREND AGENT")
        st.caption("SK의 최신 경영 전략과 IT 트렌드를 한번에 분석할 수 있다면?")

    _inject_css()
    ensure_state()
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    render_sidebar()

    example = (
        "예시:\n"
        "1) SK 텔레콤의 26년 기술 전략과 그 전략이 최신의 AI기술에 얼마나 부합해?\n"
        "2) SK AX의 26년 경영전략은?\n"
        "3) 26년 새롭게 떠오르는 생성형AI의 신규 모델은?"
    )
    question = st.text_area(
        "질문을 입력하세요",
        height=140,
        key="question_input",
        placeholder=example,
    )

    if st.button("실행", type="primary") and question:
        render_main(question)
    else:
        st.info("예시: 2026년 SK 텔레콤에서 새롭게 시도하는 기술전략을 분석하고 최신 IT 트렌드 중 반영 안된 기술을 알려줘.")
