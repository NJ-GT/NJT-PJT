# Dashboard RAG

서울시 관광 지역 내 숙박시설 화재위험 대시보드 설명용 RAG입니다.

## 구성

- `documents/`: 대시보드, 변수, 데이터 출처, 공간모형, 도로망 해석 문서
- `rag_core.py`: 로컬 문서 검색과 프롬프트 생성 공통 로직
- `query_codex.py`: OpenAI/Codex 계열 API 호출
- `query_claude.py`: Claude API 호출
- `server.py`: 로컬 HTTP API 서버

## 환경변수

API 키는 절대 Git에 커밋하지 않는다. `.env`와 `.streamlit/secrets.toml`은 `.gitignore`로 제외되어 있고, 저장소에는 `.env.example`만 올린다.

```powershell
$env:OPENAI_API_KEY="..."
$env:ANTHROPIC_API_KEY="..."
```

선택 모델:

```powershell
$env:OPENAI_RAG_MODEL="gpt-5.2"
$env:ANTHROPIC_RAG_MODEL="claude-sonnet-4-5"
```

## 사용 예시

검색만 확인:

```powershell
python rag/query_codex.py "마포구 대흥동이 왜 위험한지 설명해줘" --offline
```

Codex/OpenAI API:

```powershell
python rag/query_codex.py "도로망 추정거리와 유클리드 거리는 뭐가 달라?"
```

Claude API:

```powershell
python rag/query_claude.py "MGWR과 GWR 차이를 대시보드 기준으로 설명해줘"
```

로컬 API 서버:

```powershell
python rag/server.py --port 8765
```

엔드포인트:

- `POST /search`
- `POST /codex`
- `POST /claude`

요청 예시:

```json
{"question": "최종위험점수_new는 무엇을 의미해?"}
```
