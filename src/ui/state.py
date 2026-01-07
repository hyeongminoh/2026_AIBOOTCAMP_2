from __future__ import annotations

import streamlit as st


def ensure_state():
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts: {"q": ..., "a": ...}
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = []  # list of {"question":..., "answer":...}