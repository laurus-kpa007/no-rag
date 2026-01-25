# Advanced Multi-Mode RAG Bot 구현 계획

사용자가 요청한 4가지 검색 모드(파일, 벡터, 키워드, 하이브리드)를 지원하고, Ollama의 역할을 세분화(임베딩, 리랭크, 챗)하여 설정할 수 있는 고도화된 봇을 개발합니다.

## User Review Required
> [!IMPORTANT]
> - **추가 라이브러리**: `chromadb` (벡터 저장소), `rank_bm25` (키워드 검색) 설치가 필요합니다.
> - **Rerank 속도**: Ollama LLM을 리랭커로 사용 시(GenAI Reranking), 문서마다 LLM 호출이 발생하여 속도가 느릴 수 있습니다.
> - **파일 분할(Chunking)**: RAG를 위해서는 문서를 잘게 쪼개는 Chunking 로직이 필수적으로 추가됩니다.

## Proposed Changes

### Project Root (`d:/Python/no-rag/`)

#### [NEW] [advanced_rag_bot.py](file:///d:/Python/no-rag/advanced_rag_bot.py)
단일 파일에 모든 로직을 담되, 클래스 구조로 깔끔하게 분리합니다.

1.  **Config Class**:
    - `OLLAMA_CHAT_HOST`, `OLLAMA_EMBED_HOST`
    - `MODEL_CHAT`, `MODEL_EMBED`, `MODEL_RERANK`
    - `SEARCH_MODE` (1~4)
    - `NUM_CTX`

2.  **DocumentProcessor**:
    - `load_document(path)`: 기존 로직 재사용.
    - `chunk_text(text, size=1000)`: 텍스트를 겹치게(overlap) 분할.

3.  **Retriever Classes**:
    - `VectorRetriever`: `chromadb` 사용. Ollama Embedding API 호출.
    - `KeywordRetriever`: `rank_bm25` 사용. 텍스트 토크나이징.
    - `HybridRetriever`: 두 결과 합치기 (Weighted Sum 또는 RRF).

4.  **Reranker**:
    - `rate_documents(query, docs)`: LLM에게 "이 문서가 질문과 관련이 있나요? 0~10점" 프롬프트 전송 후 정렬.

5.  **Main Loop**:
    - 시작 시 모드 선택 (1. File, 2. Vector, 3. Keyword, 4. Hybrid)
    - 설정 확인 및 변경 메뉴 제공.
    - 질문 -> 검색 -> (리랭크) -> 답변 생성.

## Verification Plan

### Automated/Manual Tests
1.  **Dependencies**: `pip install chromadb rank_bm25 ollama python-docx`
2.  **Mode 1 (File)**: 기존과 동일하게 작동하는지 확인.
3.  **Mode 2 (Vector)**: 질문과 의미가 유사한 청크가 검색되는지 확인.
4.  **Mode 3 (Keyword)**: 특정 단어가 포함된 청크가 검색되는지 확인.
5.  **Mode 4 (Hybrid)**: 두 결과가 섞여서 나오고, Rerank 과정을 거치는지 확인.
