# No-RAG & Advanced Metadata-Driven Hybrid RAG Bot

Ollama 로컬 LLM을 활용한 **문서 기반 Q&A 봇** 프로젝트입니다.
벡터 DB 없이 문서 전체를 분석하는 **No-RAG** 방식과, 2026년 최신 **Metadata-Driven Hybrid RAG** 기술을 적용한 **Advanced RAG** 방식을 모두 지원합니다.

## ✨ 주요 특징

### 1. No-RAG Bot (`no_rag_bot.py`)
- **Full Context Stuffing**: 문서를 쪼개지 않고 통째로 프롬프트에 넣어 분석합니다.
- **최고의 정확도**: 100~200페이지 이내의 문서라면 RAG보다 훨씬 정확한 답변을 제공합니다.
- **Context Window 제어**: 32K~128K 등 LLM의 컨텍스트 한계까지 최대한 활용 가능.

### 2. Advanced RAG Bot (`advanced_rag_bot.py`) ⭐ NEW
2026년 최신 RAG 연구 트렌드를 반영한 프로덕션급 시스템:

#### 🆕 Metadata-Driven Query Correction (2026 최신 트렌드)
- **벡터 검색에 의존하지 않는 질의 교정**: 문서 인덱싱 시 도메인, 키워드, 전문 용어를 LLM으로 추출
- **31% 성능 향상**: Metadata-driven RAG 방식으로 검색 품질 대폭 개선 ([Utilizing Metadata for Better RAG, 2026](https://arxiv.org/html/2601.11863v1))
- **벡터 검색 실패에 강건함**: 초기 검색이 실패해도 메타데이터로 질의를 정확하게 교정

#### 🧠 Intelligent Query Type Detection
- **LLM 기반 질문 유형 자동 분류**: 질의 교정과 동시에 질문 유형(SEARCH/SUMMARY/COMPARE/LIST) 판단
- **컨텍스트 인식 라우팅**: 질문 유형에 따라 최적의 검색 전략 자동 선택
  - **SEARCH**: 하이브리드 검색 (Vector + BM25)
  - **SUMMARY**: 사전 생성된 요약 캐시 활용
  - **COMPARE**: 엔티티별 개별 검색 후 병합
  - **LIST**: 확장 검색 (TOP_K × 4)

#### 🔍 Multi-Mode Hybrid Search
1. **File Search**: 문서 전체 검색 (No-RAG 방식)
2. **Vector Search**: 의미 기반 검색 (ChromaDB + Ollama bge-m3)
3. **Keyword Search**: BM25 키워드 매칭 검색
4. **Hybrid Search**: Vector + Keyword 병합 후 **LLM Reranking**
5. **Auto Mode (Query Router)**: 질문 유형에 따라 자동으로 최적 검색 ★ 추천

#### 📄 Deep Document Extraction
- **Deep XML Extraction**: `.docx` 파일 내부의 **텍스트 상자, 표, 도형, 머리글/바닥글**에 숨겨진 모든 텍스트를 완벽하게 추출
- **debug_extracted.txt**: 추출된 텍스트를 파일로 저장하여 검증 가능

#### ⚡ Pre-Summarization Cache
- **인덱싱 시 사전 요약 생성**: 문서가 큰 경우 계층적 요약을 미리 생성하여 캐싱
- **빠른 요약 응답**: SUMMARY 질문 시 즉시 응답 (실시간 요약 불필요)

---

## 🚀 설치 방법

### 1. 필수 요구사항
- **Python 3.10+**
- **Ollama**: 로컬 LLM 실행을 위해 설치되어 있어야 합니다. ([설치 링크](https://ollama.com/))

### 2. 패키지 설치
```bash
git clone https://github.com/laurus-kpa007/no-rag.git
cd no-rag
pip install -r requirements.txt
```

### 3. Ollama 모델 다운로드
```bash
# 임베딩 모델
ollama pull bge-m3

# 채팅 모델 (12B 권장, 4B도 가능)
ollama pull gemma3:12b
# 또는
ollama pull gemma3:4b
```

---

## 💻 사용 방법

### 기본 실행
```bash
# 단순형 (작은 문서 추천)
python no_rag_bot.py "내문서.docx"

# 고급형 (대용량 문서 추천, 메타데이터 기반 하이브리드 검색)
python advanced_rag_bot.py "내문서.docx"
```

### Advanced Bot 검색 모드 선택
프로그램 실행 후 원하는 모드를 선택하세요:

```text
========================================
 1. 파일 전체 검색 (No-RAG, Context Stuffing)
 2. 벡터 검색 (Vector Store)
 3. 키워드 검색 (BM25)
 4. 하이브리드 검색 (Hybrid + Rerank)
 5. 자동 모드 (Query Router) ★ 추천
 q. 종료
========================================
```

#### 모드별 특징
- **모드 1 (파일 전체)**: 소규모 문서에 최적, 가장 정확
- **모드 2 (벡터)**: 의미 기반 검색, 동의어 매칭 강점
- **모드 3 (키워드)**: 정확한 용어 매칭, 고유명사 검색 강점
- **모드 4 (하이브리드)**: 벡터 + 키워드 병합 후 LLM으로 재평가
- **모드 5 (자동) ⭐**: LLM이 질문 유형을 자동 분석하여 최적 검색 전략 선택

---

## ⚙️ 설정 (Configuration)

`advanced_rag_bot.py` 상단의 `Config` 클래스에서 모델과 파라미터를 변경할 수 있습니다.

```python
class Config:
    DOC_PATH = 'data.docx'

    # Ollama 서버 설정
    OLLAMA_HOST = 'http://localhost:11434'

    # 모델 설정
    MODEL_CHAT = 'gemma3:12b'        # 답변 생성 모델
    MODEL_EMBED = 'bge-m3'           # 임베딩 모델 (768차원)
    MODEL_RERANK = 'gemma3:12b'      # 리랭킹 모델

    # 검색 설정
    CHUNK_SIZE = 500                 # 청크 크기 (글자 수)
    CHUNK_OVERLAP = 50               # 청크 겹침 크기
    TOP_K = 5                        # 검색시 가져올 문서 수
    NUM_CTX = 32768                  # LLM 컨텍스트 윈도우

    # Query Router 설정
    SUMMARY_CHUNK_SIZE = 3000        # 계층적 요약 시 청크 크기
    MAX_CONTEXT_RATIO = 0.7          # 전체 문서 투입 가능 비율
    PRE_SUMMARIZE = True             # 인덱싱 시 사전 요약 생성 여부
```

---

## 🏗️ 아키텍처

### 시스템 구조

```
사용자 질문
    ↓
[메타데이터 기반 질의 교정 + 유형 분석]  ← 2026 최신 트렌드
    ↓
[Query Router: 질문 유형별 최적 검색]
    ├─ SEARCH → Hybrid Search (Vector + BM25) + Reranking
    ├─ SUMMARY → 사전 생성 요약 캐시 사용
    ├─ COMPARE → 엔티티별 검색 + 병합
    └─ LIST → 확장 검색 (TOP_K × 4)
    ↓
[LLM 답변 생성 (Streaming)]
```

### 주요 컴포넌트
1. **MetadataStore**: 문서 도메인/키워드/전문용어 추출 및 저장
2. **VectorStore**: 의미 기반 검색 (ChromaDB + bge-m3)
3. **KeywordStore**: 키워드 기반 검색 (BM25)
4. **SummaryCache**: 사전 요약 생성 및 캐싱
5. **Query Router**: 질문 유형 분석 및 검색 전략 자동 선택

---

## 📊 성능 개선 내역

### 2026년 최신 RAG 기술 적용
- **Metadata-Driven Query Correction**: 31% 성능 향상 ([논문 링크](https://arxiv.org/html/2601.11863v1))
- **Hybrid Search (BM25 + Vector)**: 2026년 업계 표준, 순수 벡터 검색 대비 정확도 향상
- **LLM-based Query Type Detection**: 단일 LLM 호출로 교정 + 유형 분석 동시 수행 (비용 절감)
- **Pre-Retrieval Optimization**: 검색 전 질의 최적화로 검색 실패율 감소

---

## 📖 문서

- [ARCHITECTURE.md](ARCHITECTURE.md): 전체 시스템 아키텍처 상세 설명
- [hybrid_rag_design.md](hybrid_rag_design.md): 하이브리드 RAG 설계 원리
- [keyword_search_explainer.md](keyword_search_explainer.md): BM25 키워드 검색 설명

---

## 🔬 기술 스택

- **LLM Framework**: Ollama (로컬 실행)
- **Chat Model**: gemma3:12b (32K context)
- **Embedding Model**: bge-m3 (768d, 다국어 지원)
- **Vector DB**: ChromaDB (인메모리)
- **Keyword Search**: BM25Okapi (rank_bm25)
- **Document Parser**: python-docx (Deep XML extraction)

---

## 📈 사용 시나리오

### No-RAG Bot 추천 상황
- 문서 크기: ~100페이지 이하
- 문서 수: 1-2개
- 요구사항: 최고 정확도, 문맥 놓치면 안됨

### Advanced RAG Bot 추천 상황
- 문서 크기: 100페이지 이상
- 문서 수: 여러 개
- 요구사항: 빠른 검색, 확장성, 다양한 질문 유형 지원

---

## 📝 라이선스
MIT License

---

## 🙏 참고 문헌

- [Utilizing Metadata for Better Retrieval-Augmented Generation (2026)](https://arxiv.org/html/2601.11863v1)
- [METADATA-DRIVEN RETRIEVAL-AUGMENTED GENERATION FOR FINANCIAL QA (2024)](https://arxiv.org/pdf/2510.24402)
- [Advanced RAG Techniques for High-Performance LLM Applications](https://neo4j.com/blog/genai/advanced-rag-techniques/)
- [Pre-Retrieval Query Optimization in RAG Systems (2026)](https://www.educative.io/courses/advanced-rag-techniques-choosing-the-right-approach/what-is-pre-retrieval-query-optimization)
