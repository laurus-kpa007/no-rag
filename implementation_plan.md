# No-RAG Ollama Q&A 봇 구현 계획

이 계획은 복잡한 RAG/Vector DB 설정을 우회하여 "Full Context Stuffing" 방식을 사용하는 문서 기반 Q&A 봇 파이썬 스크립트 작성을 상세히 기술합니다.

## 사용자 검토 필요
> [!IMPORTANT]
> 이 스크립트는 워드 문서의 **전체** 내용을 LLM의 컨텍스트 윈도우에 로드합니다. 
> - `llama3.1`, `qwen2.5` (128k 컨텍스트) 등 컨텍스트 윈도우가 큰 Ollama 모델을 사용하거나 문서 크기가 충분히 작은지 확인하세요.
> - 문서가 너무 크면 모델이 텍스트를 자르거나 처리에 실패할 수 있습니다.

## 변경 제안

### 프로젝트 루트 (`d:/Python/no-rag/`)

#### [NEW] [no_rag_bot.py](file:///d:/Python/no-rag/no_rag_bot.py)
- **Imports**: `ollama`, `docx` (python-docx), `sys`
- **설정**: 상단에 `DOC_PATH` 변수 (기본값 'data.docx').
- **함수**:
    - `extract_text_from_docx(file_path)`: .docx 파일의 모든 단락과 표를 읽어 하나의 문자열로 반환.
    - `chat_with_doc(doc_content)`: 메인 루프:
        1. 사용자 입력 수신.
        2. 종료 키워드 확인 ('q', 'exit').
        3. 프롬프트 구성: `Context:\n{doc_content}\n\nQuestion:\n{user_input}`.
        4. `stream=True`로 `ollama.chat` 호출.
        5. 청크를 표준 출력(stdout)으로 출력.
- **메인 실행**:
    - 파일 존재 여부 확인.
    - 데이터 로드.
    - 채팅 루프 시작.
- **헤더 주석**: `pip install ollama python-docx` 포함.

## 검증 계획

### 수동 검증
1.  **의존성 설치**: 사용자가 `pip install ollama python-docx` 실행.
2.  **데이터 준비**: 폴더에 `data.docx` 이름으로 워드 파일 배치.
3.  **스크립트 실행**: `python no_rag_bot.py` 실행.
4.  **상호작용 테스트**:
    -   문서 내용에 대해 질문.
    -   답변이 스트리밍되는지 확인.
    -   'q' 입력 시 종료되는지 확인.
