# 하이브리드 RAG 아키텍처 설계 (최신 기술 트렌드 기준)

단순 "파일 서치(Context Stuffing)"와 "벡터 검색(Vector Search)"을 결합하여, **정확도**와 **문맥 이해**를 모두 잡는 최신 하이브리드 RAG 설계를 제안합니다.

## 1. 아키텍처 개요
"Hybrid Search + Reranking" 패턴이 현재 업계 표준(SOTA)에 가깝습니다.

### 핵심 구성 요소
1.  **Vector Search (Semantic Search)**: 의미 기반 검색 (예: "가전제품 고장" -> "수리 방법" 문서 검색).
2.  **Keyword Search (BM25/Sparse)**: 정확한 키워드 매칭 (예: 모델명 "XG-200" -> "XG-200" 스펙 문서).
3.  **Reranking (Cross-Encoder)**: 위 두 검색 결과를 합친 후, LLM이 채점하여 다시 순위를 매김 (정확도 급상승).
4.  **Long-Context LLM**: 추출된 최상위 문서들을 여유롭게 입력받아 최종 답변 생성.

---

## 2. 기술 스택 추천 (Python 기반)

| 구분 | 추천 기술 | 설명 |
| :-- | :-- | :-- |
| **Orchestrator** | **LlamaIndex** 또는 **LangChain** | 하이브리드 검색 구현이 가장 잘 되어 있는 프레임워크. 특히 LlamaIndex가 RAG에 특화됨. |
| **Vector DB** | **ChromaDB** 또는 **Qdrant** | 로컬에서 가볍게 돌리기 좋고, Python 친화적이며 설치가 쉬움. |
| **Embedding** | **BGE-m3** 또는 **Ko-SBERT** | 한국어 성능이 뛰어난 임베딩 모델 (HuggingFace). |
| **Reranker** | **BGE-Reranker-v2-m3** | 검색 결과의 순위를 재조정하는 필수 요소. (Ollama로도 가능하지만 전용 모델이 훨씬 빠르고 정확함) |
| **Local LLM** | **Ollama** (Llama 3.1, Gemma 2) | 현재 사용 중인 스택 유지. |

---

## 3. 구현 로직 (Workflow)

```mermaid
graph TD
    A[사용자 질문] --> B{Hybrid Retriever}
    B -->|Vector Search| C[FAISS/ChromaDB]
    B -->|Keyword Search| D[BM25 Retriever]
    C & D --> E[검색 결과 병합 (Reciprocal Rank Fusion)]
    E --> F[Reranker (Cross-Encoder)]
    F -->|Top-K 문서 선정| G[Prompt Template]
    G --> H[Ollama LLM (Streaming)]
    H --> I[최종 답변]
```

## 4. 기존 코드(`no_rag_bot.py`)와의 통합 방안

지금 만드신 **Full Context Stuffing** 방식은 문서가 적을 때(책 1권 분량) 가장 정확합니다. 하지만 문서가 수백 개가 되면 비용과 속도 문제가 발생합니다.

### 단계별 발전 전략
1.  **Level 1 (현재)**: 문서 1개 통째로 넣기. (가장 정확, 속도 느림, 문서 크기 제한)
2.  **Level 2 (Chunking)**: 문서를 500자 단위로 자르고, "질문과 관련된 부분만" 가져오기.
3.  **Level 3 (Hybrid)**: "의미 검색" + "키워드 검색"을 섞어서 가져온 뒤, Reranker로 정제하여 LLM에 입력.

## 5. 결론 및 제안
**"어떤 상황에서 RAG를 도입해야 하는가?"**
- 문서가 **100페이지를 넘어가고**, 여러 개의 파일(PDF, DOCX 등)을 동시에 검색해야 한다면 **Hybrid RAG**로 넘어가야 합니다.
- 현재처럼 파일 1~2개를 정밀 분석하는 용도라면, 지금 구현하신 **Full Context Stuffing + 32K Window** 방식이 오히려 RAG보다 답변 품질이 좋을 수 있습니다. (RAG는 정보를 찾다가 놓칠 수 있기 때문)
