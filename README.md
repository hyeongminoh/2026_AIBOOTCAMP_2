# SK Tech Strategy Alignment Agent

멀티 에이전트 + RAG 조합으로 SK/SK AX 관련 전략을 글로벌 기술 트렌드와 비교·판단하는 데모 서비스입니다. AOAI(Azure OpenAI) 기반 LLM/임베딩을 사용합니다.

## 주요 기능
- LangGraph 기반 멀티 에이전트 파이프라인 (Planner → RAG → Alignment → Formatter)
- AOAI Chat/Embedding + Chroma 벡터DB RAG
- Streamlit UI 및 FastAPI 백엔드 `/ask`
- 샘플 코퍼스(SK 뉴스, 글로벌 트렌드) 제공

## 폴더 구조
- `app.py` : Streamlit UI
- `server.py` : FastAPI 엔드포인트
- `src/config.py` : 환경변수 로딩
- `src/graph.py` : LangGraph 워크플로우 정의
- `src/agents/` : 역할별 프롬프트
- `src/rag/` : 인덱싱/리트리버 유틸
- `data/*.json` : 샘플 코퍼스, `data/chroma/` : 벡터DB(생성됨)

## 사전 준비
1) Python 3.10+ 추천  
2) 가상환경 생성 후 패키지 설치
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3) AOAI 환경 변수 설정
```bash
export AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<your-key>"
export AZURE_OPENAI_CHAT_DEPLOYMENT="<chat-deployment-name>"
export AZURE_OPENAI_EMBED_DEPLOYMENT="<embedding-deployment-name>"
export AZURE_OPENAI_API_VERSION="2024-05-01-preview"
```

## 벡터DB 생성 (선택)
샘플 문서로 Chroma를 미리 생성하려면:
```bash
python -m src.rag.ingest --persist-dir data/chroma
```
생성하지 않아도 실행 시 in-memory 로 빌드합니다(최초 쿼리 시 AOAI 임베딩 호출).

## 실행 방법
- Streamlit UI
```bash
streamlit run app.py
```
- FastAPI 서버
```bash
uvicorn server:app --reload --port 8000
```
POST `/ask` 바디 예시:
```json
{ "question": "2026년 SK에서 새롭게 시도하는 기술은 무엇이며, 최신 트렌드를 반영하는가?" }
```

## 동작 플로우
1. Planner: 질문 의도·필수 리소스 판단(JSON 계획)
2. RAG: SK / 글로벌 코퍼스 별도 검색
3. Alignment: 두 컨텍스트 비교 후 기회/리스크/권고 생성
4. Formatter: 한줄 요약, 상세, 근거 기사, 다음 액션을 구조화

## 확장 아이디어
- 추가 벡터 소스(사내 데이터 레이크) 연결
- Agent-to-Agent 협업 및 함수 호출 기반 유형 분기
- 리포트/표 형태의 Structured Output
- 감사 로그 및 관측(Tracing) 추가

## 주의사항
- AOAI 키는 코드에 하드코딩하지 말고 환경 변수로만 관리하세요.
- 로컬 샘플 데이터는 데모용이며 실제 서비스 시 최신 데이터로 교체하세요.

