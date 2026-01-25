# No-RAG & Advanced Hybrid RAG Bot

Ollama 로컬 LLM을 활용한 **문서 기반 Q&A 봇** 프로젝트입니다.  
벡터 DB 없이 문서 전체를 분석하는 **No-RAG** 방식과, 최신 하이브리드 검색 기술을 적용한 **Advanced RAG** 방식을 모두 지원합니다.

## ✨ 주요 특징

### 1. No-RAG Bot (`no_rag_bot.py`)
- **Full Context Stuffing**: 문서를 쪼개지 않고 통째로 프롬프트에 넣어 분석합니다.
- **최고의 정확도**: 100~200페이지 이내의 문서라면 RAG보다 훨씬 정확한 답변을 제공합니다.
- **Context Window 제어**: 32K~128K 등 LLM의 컨텍스트 한계까지 최대한 활용 가능.

### 2. Advanced RAG Bot (`advanced_rag_bot.py`)
- **Multi-Mode Search**:
    1. **File Search**: 문서 전체 검색 (No-RAG)
    2. **Vector Search**: 의미 기반 검색 (ChromaDB + Ollama Embedding)
    3. **Keyword Search**: 단어 매칭 검색 (BM25)
    4. **Hybrid Search**: 벡터 + 키워드 검색 후 **Reranking(재순위)** 과정을 거쳐 최적의 답변 도출.
- **Deep Text Extraction**: `.docx` 파일 내부의 **텍스트 상자, 표, 도형**에 숨겨진 텍스트까지 완벽하게 추출합니다.

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

---

## 💻 사용 방법

### 기본 실행
```bash
# 단순형 (작은 문서 추천)
python no_rag_bot.py "내문서.docx"

# 고급형 (대용량 문서 추천, 하이브리드 검색)
python advanced_rag_bot.py "내문서.docx"
```

### Advanced Bot 실행 예시
프로그램 실행 후 원하는 모드를 선택하세요.
```text
 1. 파일 전체 검색 (No-RAG)
 2. 벡터 검색 (Vector Store)
 3. 키워드 검색 (BM25)
 4. 하이브리드 검색 (Hybrid + Rerank)
```
- **모드 4(Hybrid)**를 추천합니다. 질문의 키워드와 의미를 동시에 파악하고, LLM이 한 번 더 검증(Rerank)하여 답변합니다.

---

## ⚙️ 설정 (Configuration)

`advanced_rag_bot.py` 상단의 `Config` 클래스에서 모델을 변경할 수 있습니다.

```python
class Config:
    OLLAMA_HOST = 'http://localhost:11434'
    
    MODEL_CHAT = 'gemma3:4b'      # 답변 생성 모델
    MODEL_EMBED = 'nomic-embed-text' # 임베딩 모델
    MODEL_RERANK = 'gemma3:4b'    # 리랭킹 모델
    
    NUM_CTX = 32768               # 컨텍스트 윈도우 크기
```

---

## 📝 라이선스
MIT License
