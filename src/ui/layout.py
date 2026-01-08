from __future__ import annotations

import re
import os
from typing import List, Tuple

import requests
import streamlit as st

from src.ui.state import ensure_state


ICONS = {
    "planner": "🧭 Planner",
    "sk_rag": "📰 SK RAG",
    "global_rag": "🧠 IT RAG",
    "alignment_step": "⚖️ Alignment",
    "format": "🧩 Formatter",
}

LOGO_PATH = "src/ui/SKLOGO.png"
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


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


def render_main(question: str, main_col, side_col):
    with main_col:
        st.markdown(
            f"""
            <div style="padding:10px 12px; border:1px solid #e5e7eb; border-radius:8px; background:#f1f5f9; margin-bottom:10px;">
                <strong>입력한 질문</strong><br>{question}
            </div>
            """,
            unsafe_allow_html=True,
        )
        answer_placeholder = st.container()
    with side_col:
        summary_container = st.container()
        log_placeholder = st.empty()
    logs: List[Tuple[str, str]] = []

    def append_log(msg: str, kind: str = "info"):
        logs.append((kind, msg))
        colors = {
            "info": "#111827",
            "success": "#0ea65c",
            "warning": "#d97706",
            "error": "#dc2626",
        }
        html = (
            "<div style='max-height:260px;overflow:auto;padding:8px 10px;"
            "border:1px solid #eee;border-radius:8px;background:#fafafa'>"
        )
        for level, text in logs:
            color = colors.get(level, "#111827")
            html += f"<div style='color:{color};margin-bottom:4px;'>{text}</div>"
        html += "</div>"
        log_placeholder.markdown(html, unsafe_allow_html=True)

    try:
        append_log(f"API 호출: {API_BASE}/ask", "info")
        resp = requests.post(f"{API_BASE}/ask", json={"question": question}, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"/ask {resp.status_code} {resp.text}")
        data = resp.json()
        answer_text = data.get("answer", "응답이 비어 있습니다.")
        agents = data.get("agents", [])
        tools = data.get("tools", [])
        cleaned_answer = re.sub(r"\n{3,}", "\n\n", answer_text)

        if agents:
            agent_flow = " → ".join([ICONS.get(a, a) for a in agents])
            summary_container.success(f"에이전트 흐름: {agent_flow}")
        if tools:
            tool_lines = "\n".join([f"- {t}" for t in tools])
            summary_container.info(f"툴 호출:\n{tool_lines}")

        answer_placeholder.markdown(
            f"""
            <div class="answer-card">
                <div class="answer-title">답변</div>
                <div class="answer-body">{cleaned_answer}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.history.append({"q": question, "a": answer_text})
        st.session_state.clear_input = True
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
        col_title, col_home = st.columns([6, 1])
        with col_title:
            st.title("SK TECH & IT TREND AGENT")
            st.caption("SK의 최신 경영 전략과 IT 트렌드를 한번에 분석할 수 있다면?")
        with col_home:
            if st.button("홈 새로고침"):
                st.experimental_rerun()

    _inject_css()
    ensure_state()
    if st.session_state.get("clear_input"):
        st.session_state.question_input = ""
        st.session_state.clear_input = False
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    render_sidebar()

    example = (
        "예시:\n"
        "1) SK 텔레콤의 26년 기술 전략과 그 전략이 최신의 AI기술에 얼마나 부합해?\n"
        "2) SK AX의 26년 경영전략은?\n"
        "3) 26년 새롭게 떠오르는 생성형AI의 신규 모델은?"
    )
    col_main, col_side = st.columns([2, 1])
    with col_main:
        question = st.text_area(
            "질문을 입력하세요",
            height=140,
            key="question_input",
            placeholder=example,
        )
        run_clicked = st.button("실행", type="primary")
    with col_side:
        st.empty()  # 사이드 공간 확보

    if run_clicked and question:
        render_main(question, col_main, col_side)
    else:
        st.info("예시: 2026년 SK 텔레콤에서 새롭게 시도하는 기술전략을 분석하고 최신 IT 트렌드 중 반영 안된 기술을 알려줘.")
