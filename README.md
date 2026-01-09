# SK TECH & IT TREND AGENT

멀티 에이전트 + RAG로 SK/SK AX 관련 전략을 최신 IT 트렌드와 비교·판단하는 데모 서비스입니다. AOAI(Azure OpenAI) 기반 LLM/임베딩을 기본으로 사용하며, LangGraph/ToolCalling/ReAct/메모리를 포함합니다.

## 주요 기능
- LangGraph 멀티에이전트: Planner → ReAct Refine → SK RAG → IT RAG → Alignment → Formatter (조건부 분기)
- Tool Calling: classify_query, search_sk_news, search_global_it(IT), score_alignment
- ReAct 보조 검색: Planner 후 추가 검색 질의 정제/보강
- Memory: 최근 Q/A를 Formatter에 주입
- RAG: FAISS + AOAI 임베딩, SK→IT 순으로 검색하며 SK 결과를 IT 검색 힌트로 전달
- UI/UX: Streamlit 실시간 배너(에이전트/툴 로그), 답변 카드, 질문 히스토리, 로고 적용
- API: FastAPI `/ask`

## 폴더 구조
- `app.py` : Streamlit 엔트리(UI가 API 호출)
- `server.py` : FastAPI 엔드포인트
- `src/graph.py` : LangGraph 워크플로우
- `src/agents/` : 역할별 프롬프트
- `src/rag/` : ingest/retriever/embedding 유틸
- `src/ui/` : UI 렌더링, 상태/로고
- `data/*.json` : 샘플 코퍼스, `data/faiss/` : FAISS 인덱스(생성됨)

## 사전 준비 (Windows PowerShell 예시)
```powershell
Set-Location 'C:\Users\10377\Desktop\2026_AIBOOTCAMP_2\2026_AIBOOTCAMP_2'
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 환경 변수 (AOAI)
```powershell
$env:AZURE_OPENAI_ENDPOINT="https://<your-endpoint>.openai.azure.com"
$env:AZURE_OPENAI_API_KEY="<your-key>"
$env:AZURE_OPENAI_CHAT_DEPLOYMENT="<chat-deployment-name>"
$env:AZURE_OPENAI_EMBED_DEPLOYMENT="<embedding-deployment-name>"
$env:AZURE_OPENAI_API_VERSION="2024-05-01-preview"
```

## 벡터DB 생성 (선택)
샘플 문서로 FAISS를 미리 생성:
```powershell
python -m src.rag.ingest --persist-dir data/faiss
```
없으면 첫 질의 시 in-memory로 빌드(임베딩 호출 발생).

## 실행 방법
1) FastAPI 서버 실행 (포트 8001):
```powershell
uvicorn server:app --host 0.0.0.0 --port 8001
```
2) Streamlit 실행 (API_BASE를 8001로 맞춤):
```powershell
$env:API_BASE_URL="http://localhost:8001"
streamlit run app.py --server.address=0.0.0.0 --server.port 8516
```
3) 접속/테스트
- UI: http://localhost:8516
- 헬스: http://localhost:8001/health
- API: `POST /ask`
```bash
curl -X POST http://localhost:8001/ask -H "Content-Type: application/json" -d "{\"question\":\"테스트\"}"
```

## 동작 플로우 (간략)
1) Planner: LLM 기반 JSON 플랜(+툴 fallback) → 의도/필수 리소스 결정  
2) ReAct Refine: 필요 시 추가 검색 툴 호출  
3) SK RAG → IT RAG: SK 검색 후 결과를 IT 검색 힌트로 활용  
4) Alignment: 정합성 분석 + 점수 산출  
5) Formatter: 답변 카드, 근거, 점수, 다음 액션 출력 (최근 히스토리 포함)

## 샘플 질문
- SK 텔레콤의 26년 기술 전략과 최신 AI 기술 부합도?
- SK AX의 26년 경영전략은?
- 26년 새롭게 떠오르는 생성형 AI 신규 모델은?

## 주의/참고
- FAISS 사용: Py3.12 환경에서 chroma 빌드 이슈가 있어 FAISS로 전환
- AOAI 키는 코드에 하드코딩 금지, 환경변수만 사용
- 샘플 데이터는 데모용, 실제 서비스 시 최신 데이터로 교체

## Docker 배포 (API+UI)
### 단일 실행
- FastAPI(8001): `docker build -f Dockerfile.api -t sk-agent-api .` → `docker run -p 8001:8001 sk-agent-api`
- Streamlit(8516): `docker build -f Dockerfile.ui -t sk-agent-ui .` → `docker run -p 8516:8516 sk-agent-ui`

### docker-compose (권장)
```bash
docker compose build
docker compose up -d
```
- UI: http://localhost:8516
- 헬스: http://localhost:8001/health

## 향후 개선 아이디어
- RAG 품질 개선: 재랭커/하이브리드/날짜 스코어링(date_num) 등
- 더 긴 메모리/대화 맥락 주입 및 Alignment에도 메모리 반영

