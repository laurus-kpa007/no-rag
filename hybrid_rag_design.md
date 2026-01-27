# Metadata-Driven Hybrid RAG 아키텍처 (2026 최신 트렌드 기반)

이 문서는 본 프로젝트에 구현된 **Metadata-Driven Hybrid RAG** 시스템의 설계 원리와 2026년 최신 RAG 연구 트렌드를 설명합니다.

---

## 📌 2026년 RAG 트렌드 요약

### 주요 변화
1. **Metadata-Driven Query Correction**: 벡터 검색 없이 메타데이터로 질의 교정 (31% 성능 향상)
2. **Hybrid Search 표준화**: BM25 + Vector Search 조합이 업계 표준
3. **Pre-Retrieval Optimization**: 검색 전 질의 최적화가 핵심
4. **Query-Aware Routing**: 질문 유형에 따른 동적 검색 전략

---

## 1. 아키텍처 개요

본 프로젝트는 다음 핵심 기술을 결합합니다:

```
[문서 인덱싱]
  ↓
1. Metadata Extraction (도메인, 키워드, 전문용어)  ← NEW (2026)
2. Vector Indexing (의미 기반)
3. Keyword Indexing (BM25)
4. Pre-Summarization (요약 캐싱)
  ↓
[질의 처리]
  ↓
1. Metadata-based Query Correction + Type Detection  ← NEW (2026)
2. Query Router (질문 유형별 검색 전략)
  ↓
[검색 실행]
  ↓
- SEARCH: Hybrid Search (Vector + BM25) + Reranking
- SUMMARY: Pre-generated Summary Cache
- COMPARE: Entity-wise Search + Merge
- LIST: Extended Search (TOP_K × 4)
  ↓
[LLM 답변 생성]
```

---

## 2. 핵심 구성 요소

### 2.1 Metadata Store (2026 최신 기술)

**문제점 (기존 방식)**:
```python
# 기존: 벡터 검색 결과로 질의 교정
pre_search_docs = vector_store.search(original_query, top_k=3)
query = correct_query(original_query, context=pre_search_docs)
# ❌ 벡터 검색이 실패하면 → 잘못된 문서 → 잘못된 교정
```

**해결 (Metadata-Driven)**:
```python
# 인덱싱 시: 문서에서 메타데이터 추출
metadata_store.extract_metadata(full_doc)
# → domain: "의료기기 제품 사양서"
# → keywords: ["출력", "전압", "전류", "정격"]
# → technical_terms: ["XG-200", "CE마크", "IEC60601"]

# 질의 시: 메타데이터 기반 교정 (벡터 검색 없음)
query, query_type = correct_query_with_metadata(original_query, metadata_store)
# ✅ 벡터 검색 실패와 무관하게 정확한 교정
```

**장점**:
- 벡터 검색 실패에 강건함
- 도메인 특화 용어 정확히 매칭
- 31% 성능 향상 (연구 결과)

### 2.2 Hybrid Search (Vector + BM25)

**Vector Search (의미 기반)**:
- 임베딩 모델: `bge-m3` (768차원, 다국어)
- 장점: 동의어 매칭 ("자동차" ≈ "차량")
- 단점: 정확한 키워드 미스 ("XG-200" 약함)

**Keyword Search (BM25)**:
- 알고리즘: BM25Okapi
- 장점: 정확한 문자열 매칭 ("XG-200" 강함)
- 단점: 동의어 미스 ("비용" ≠ "가격")

**Hybrid = 둘의 장점 결합**:
```python
# 병렬 검색
vec_docs = vector_store.search(query, top_k=5)
key_docs = keyword_store.search(query, top_k=5)

# 결과 병합 (합집합)
combined_docs = list(set(vec_docs + key_docs))

# LLM Reranking (관련성 재평가)
final_docs = rerank_documents(query, combined_docs)
```

### 2.3 Query Router (질문 유형 인식)

**LLM이 질문 유형을 자동 분류**:
```python
# 질의 교정 + 유형 분석 (한 번의 LLM 호출)
query, query_type = correct_query_with_metadata(original_query, metadata_store)

# 유형별 최적 검색 전략
if query_type == QueryType.SEARCH:
    # 하이브리드 검색 (Vector + BM25 + Reranking)
elif query_type == QueryType.SUMMARY:
    # 사전 생성된 요약 캐시 사용 (빠름)
elif query_type == QueryType.COMPARE:
    # 엔티티별 검색 후 병합
elif query_type == QueryType.LIST:
    # 확장 검색 (TOP_K × 4)
```

**질문 유형 예시**:
- SEARCH: "재택근무 승인 절차는?" → 특정 정보 검색
- SUMMARY: "문서 요약해줘" → 전체 문서 요약
- COMPARE: "A와 B의 차이점은?" → 비교/대조
- LIST: "모든 제품 목록은?" → 전체 나열

### 2.4 Pre-Summarization Cache

**문제점**: 큰 문서의 요약 질문 시 매번 실시간 요약 (느림)

**해결**:
```python
# 인덱싱 시 사전 요약 생성
summary_cache.generate(full_doc)
# → 계층적 요약 (섹션별 요약 → 통합 요약)

# 질의 시 즉시 응답
cached_summary = summary_cache.get_summary()
# ✅ 실시간 요약 불필요, 빠른 응답
```

### 2.5 LLM Reranking

**하이브리드 검색 후 품질 보장**:
```python
for doc in combined_docs:
    prompt = f"""
    이 문서가 질문과 관련있나요?
    질문: {query}
    문서: {doc[:500]}...

    Yes/No로만 답하세요.
    """
    answer = llm.chat(prompt)
    if "yes" in answer.lower():
        keep_docs.append(doc)
```

---

## 3. 구현된 검색 모드

### 모드 1: 파일 전체 검색 (No-RAG)
```python
context = full_document
# 가장 정확, 소규모 문서에 최적
```

### 모드 2: 벡터 검색
```python
docs = vector_store.search(query, top_k=5)
# 의미 기반, 동의어 매칭 강점
```

### 모드 3: 키워드 검색 (BM25)
```python
docs = keyword_store.search(query, top_k=5)
# 정확한 키워드 매칭, 고유명사 강점
```

### 모드 4: 하이브리드 검색
```python
vec_docs = vector_store.search(query, top_k=5)
key_docs = keyword_store.search(query, top_k=5)
combined = merge_and_deduplicate(vec_docs, key_docs)
final_docs = rerank_documents(query, combined)
# 최고 품질, LLM으로 재평가
```

### 모드 5: 자동 모드 (Query Router) ⭐ 추천
```python
# LLM이 질문 유형 자동 분석
query, query_type = correct_query_with_metadata(original_query, metadata_store)

# 유형별 최적 검색 전략 자동 선택
if query_type == QueryType.SEARCH:
    return hybrid_search(query)
elif query_type == QueryType.SUMMARY:
    return summary_cache.get_summary()
# ...
```

---

## 4. 기술 스택 선택 이유

| 구분 | 선택 | 이유 |
|------|------|------|
| **Vector DB** | ChromaDB | 로컬 설치 간편, Python 친화적, 인메모리 모드 지원 |
| **Embedding** | bge-m3 | 한국어 성능 우수, 768차원, 다국어 지원, Ollama 통합 |
| **Keyword** | BM25Okapi | 가볍고 빠름, 별도 서버 불필요, 정확한 키워드 매칭 |
| **LLM** | Ollama (gemma3:12b) | 로컬 실행, 32K 컨텍스트, 프라이버시 보장 |
| **Reranker** | LLM 직접 호출 | 별도 모델 불필요, Ollama로 통합 |

---

## 5. 성능 최적화 전략

### 5.1 인덱싱 최적화
```python
# 병렬 인덱싱
vector_store.add_documents(chunks)  # 임베딩 생성 (느림)
keyword_store.add_documents(chunks)  # 토큰화만 (빠름)
```

### 5.2 검색 최적화
```python
# 메타데이터 기반 교정 (벡터 검색 불필요)
query, query_type = correct_query_with_metadata(original_query, metadata_store)

# 유형별 검색 전략 (불필요한 검색 제거)
if query_type == QueryType.SUMMARY:
    return cached_summary  # 검색 없이 즉시 반환
```

### 5.3 응답 최적화
```python
# 스트리밍 응답 (실시간 출력)
for chunk in llm.chat(prompt, stream=True):
    print(chunk, end="", flush=True)
```

---

## 6. 2026년 RAG 트렌드 적용 내역

### ✅ 적용된 최신 기술

1. **Metadata-Driven Query Correction**
   - 출처: [Utilizing Metadata for Better RAG (2026)](https://arxiv.org/html/2601.11863v1)
   - 성능: 31% 향상
   - 구현: `MetadataStore` 클래스

2. **Hybrid Search (BM25 + Vector)**
   - 출처: [Advanced RAG Techniques (2026)](https://neo4j.com/blog/genai/advanced-rag-techniques/)
   - 업계 표준화
   - 구현: `VectorStore` + `KeywordStore`

3. **Pre-Retrieval Query Optimization**
   - 출처: [Pre-Retrieval Optimization Guide (2026)](https://www.educative.io/courses/advanced-rag-techniques-choosing-the-right-approach/what-is-pre-retrieval-query-optimization)
   - 검색 전 질의 최적화
   - 구현: `correct_query_with_metadata()`

4. **Query-Aware Routing**
   - 질문 유형별 동적 검색 전략
   - 구현: `QueryRouter` + `QueryType`

5. **LLM Reranking**
   - 하이브리드 검색 결과 재평가
   - 구현: `rerank_documents()`

---

## 7. 기존 No-RAG와의 비교

| 특성 | No-RAG Bot | Advanced RAG Bot |
|------|------------|------------------|
| **정확도** | ⭐⭐⭐⭐⭐ (최고) | ⭐⭐⭐⭐ |
| **속도** | ⭐⭐ (느림) | ⭐⭐⭐⭐ (빠름) |
| **문서 크기** | ~100페이지 | 무제한 |
| **확장성** | ❌ | ✅ |
| **메모리** | 높음 | 낮음 |
| **복잡도** | 단순 | 복잡 |

---

## 8. 사용 권장 사항

### No-RAG Bot 추천 상황
- 문서 크기: ~100페이지 이하
- 문서 수: 1-2개
- 요구사항: 최고 정확도, 문맥 놓치면 안됨
- 예: 계약서 정밀 분석, 법률 문서 검토

### Advanced RAG Bot 추천 상황
- 문서 크기: 100페이지 이상
- 문서 수: 여러 개
- 요구사항: 빠른 검색, 확장성, 다양한 질문 유형
- 예: 기술 매뉴얼 검색, 대규모 문서 데이터베이스

---

## 9. 향후 개선 방향

### 단기 (구현 예정)
- [ ] 한국어 형태소 분석기 통합 (Mecab)
- [ ] 문서 필터링 (시간, 출처, 카테고리)
- [ ] 멀티턴 대화 지원 (대화 히스토리)

### 중기 (연구 중)
- [ ] GraphRAG (지식 그래프 기반 RAG)
- [ ] Agentic RAG (자율 에이전트 RAG)
- [ ] Self-Corrective RAG (자가 수정)

### 장기 (트렌드 추적)
- [ ] Late Interaction Models (ColBERT)
- [ ] RAPTOR (계층적 컨텍스트)
- [ ] Contextual Retrieval (문맥 인식 검색)

---

## 10. 참고 문헌

### 핵심 논문
1. [Utilizing Metadata for Better Retrieval-Augmented Generation (2026)](https://arxiv.org/html/2601.11863v1) - 본 프로젝트의 핵심 기술
2. [METADATA-DRIVEN RAG FOR FINANCIAL QA (2024)](https://arxiv.org/pdf/2510.24402)
3. [Query Rewriting for Retrieval-Augmented LLMs (2023)](https://arxiv.org/abs/2305.14283)

### 최신 기술 가이드
1. [Advanced RAG Techniques (Neo4j, 2026)](https://neo4j.com/blog/genai/advanced-rag-techniques/)
2. [Pre-Retrieval Query Optimization (Educative, 2026)](https://www.educative.io/courses/advanced-rag-techniques-choosing-the-right-approach/what-is-pre-retrieval-query-optimization)
3. [Beyond Basic RAG: Query-Aware Systems (2026)](https://ragaboutit.com/beyond-basic-rag-building-query-aware-hybrid-retrieval-systems-that-scale/)

### 업계 동향
1. [RAG in 2026: Practical Blueprint](https://dev.to/suraj_khaitan_f893c243958/-rag-in-2026-a-practical-blueprint-for-retrieval-augmented-generation-16pp)
2. [RAG at Scale (Redis, 2026)](https://redis.io/blog/rag-at-scale/)

---

## 결론

본 프로젝트는 **2026년 최신 RAG 연구 트렌드**를 실제 구현한 프로덕션급 시스템입니다:

1. ✅ **Metadata-Driven**: 벡터 검색 실패에 강건한 질의 교정
2. ✅ **Hybrid Search**: BM25 + Vector의 장점 결합
3. ✅ **Query Router**: 질문 유형별 최적 검색 전략
4. ✅ **Pre-Optimization**: 검색 전 질의 최적화
5. ✅ **LLM Reranking**: 검색 결과 품질 보장

이는 단순한 RAG 구현이 아닌, **2026년 업계 표준을 따르는 최신 시스템**입니다.
