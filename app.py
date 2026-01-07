import os

import streamlit as st

from src.graph import run_agent


st.set_page_config(page_title="SK Tech Strategy Alignment", layout="wide")
st.title("SK Tech Strategy Alignment Agent")
st.caption("AOAI + LangGraph 기반 멀티 에이전트 데모")

with st.sidebar:
    st.header("환경 변수")
    st.write("AZURE_OPENAI_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT", "미설정"))
    st.write("AZURE_OPENAI_CHAT_DEPLOYMENT", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "미설정"))
    st.write(
        "AZURE_OPENAI_EMBED_DEPLOYMENT", os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT", "미설정")
    )
    st.write("AZURE_OPENAI_API_VERSION", os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"))

question = st.text_area("질문을 입력하세요", height=120)
if st.button("실행", type="primary") and question:
    with st.spinner("에이전트가 분석 중..."):
        try:
            answer = run_agent(question)
            st.success("완료")
            st.markdown(answer)
        except Exception as e:
            st.error(f"오류 발생: {e}")
else:
    st.info("예시: 2026년 SK에서 새롭게 시도하는 기술은 무엇이며, 최신 트렌드를 반영하는가?")

