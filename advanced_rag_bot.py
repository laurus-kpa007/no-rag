# 필요한 라이브러리 설치:
# pip install ollama chromadb rank_bm25 python-docx

import os
import sys
import ollama
import chromadb
from rank_bm25 import BM25Okapi
from docx import Document
from chromadb.utils import embedding_functions

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
class Config:
    DOC_PATH = 'data.docx'

    # Ollama 서버 설정 (필요시 수정)
    OLLAMA_HOST = 'http://localhost:11434'

    # 모델 설정
    MODEL_CHAT = 'gemma3:12b'      # 답변 생성용 LLM
    MODEL_EMBED = 'bge-m3' # 임베딩용 모델 (없으면 'ollama pull nomic-embed-text')
    MODEL_RERANK = 'gemma3:12b'    # 리랭킹용 LLM (가벼운 모델 권장)

    # 검색 설정
    CHUNK_SIZE = 500              # 청크 크기 (글자 수)
    CHUNK_OVERLAP = 50            # 청크 겹침 크기
    TOP_K = 5                     # 검색시 가져올 문서 수
    NUM_CTX = 32768               # LLM 컨텍스트 윈도우

    # Query Router 설정
    SUMMARY_CHUNK_SIZE = 3000     # 계층적 요약 시 청크 크기
    MAX_CONTEXT_RATIO = 0.7       # 전체 문서 투입 가능 비율 (NUM_CTX 대비)
    PRE_SUMMARIZE = True          # 인덱싱 시 사전 요약 생성 여부

# Ollama 클라이언트 (원격 호스트 지원)
def get_ollama_client():
    """OLLAMA_HOST 설정을 사용하는 클라이언트 반환"""
    return ollama.Client(host=Config.OLLAMA_HOST)

# ==========================================
# 2. 문서 로더 & 청킹 (Loader & Chunking)
# ==========================================
def load_document(file_path):
    if not os.path.exists(file_path):
        return None
    ext = os.path.splitext(file_path)[1].lower()
    text = ""
    try:
        if ext == '.docx':
            doc = Document(file_path)
            full_text = []

            # [강력한 추출 방법] 
            # 문서의 구조(단락, 표)를 따르지 않고, 내부 XML에서 모든 텍스트 태그(<w:t>)를 검색합니다.
            # 이 방법은 텍스트 상자, 도형, 표 등 모든 위치의 텍스트를 가져올 수 있습니다.
            
            # 1. Main Body Text
            from docx.oxml.ns import qn
            
            def get_all_text_from_element(element):
                text_list = []
                # w:t (text) 태그를 모두 찾음
                for t in element.findall('.//' + qn('w:t')):
                     if t.text:
                         text_list.append(t.text)
                return text_list

            # 문서 본문(Body) 전체 스캔
            body_texts = get_all_text_from_element(doc.element.body)
            full_text.extend(body_texts)

            text = "\n".join(full_text)
            
            # 너무 붙어서 나오면 가독성이 떨어지므로, 적절히 줄바꿈 보정이 필요할 수 있으나
            # 우선 누락 방지가 최우선이므로 simple join 사용.

        elif ext in ['.md', '.txt']:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # [DEBUG] 추출된 텍스트 저장 (사용자 확인용)
        with open("debug_extracted.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[System] 추출된 텍스트를 'debug_extracted.txt'에 저장했습니다. 내용이 맞는지 확인해보세요.")
            
    except Exception as e:
        print(f"문서 로드 오류: {e}")
        return None
    return text

def chunk_text(text, size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# ==========================================
# 3. 메타데이터 저장소 (Metadata Store)
# ==========================================
class MetadataStore:
    """
    문서 인덱싱 시 추출한 메타데이터를 저장합니다.
    - 도메인/주제
    - 주요 키워드
    - 전문 용어
    """
    def __init__(self):
        self.domain = ""
        self.keywords = []
        self.technical_terms = []
        self.is_ready = False

    def extract_metadata(self, full_doc):
        """
        문서 전체에서 메타데이터를 추출합니다.
        """
        print("\n[Metadata] 문서 메타데이터 추출 중...")

        # 문서가 너무 크면 앞부분만 샘플링
        sample = full_doc[:3000] if len(full_doc) > 3000 else full_doc

        prompt = f"""다음 문서를 분석하여 메타데이터를 추출하세요.

[문서 샘플]
{sample}

다음 형식으로만 출력하세요:
도메인: [문서의 주제/분야를 한 문장으로]
키워드: [중요 키워드 5-10개, 쉼표로 구분]
전문용어: [전문 용어/고유명사 5-10개, 쉼표로 구분]

예시:
도메인: 의료기기 제품 사양서
키워드: 출력, 전압, 전류, 정격, 안전규격, 인증
전문용어: XG-200, CE마크, IEC60601, 절연저항, 누설전류"""

        try:
            client = get_ollama_client()
            res = client.chat(model=Config.MODEL_CHAT, messages=[{'role': 'user', 'content': prompt}])
            content = res['message']['content'].strip()

            # 파싱
            for line in content.split('\n'):
                if line.startswith('도메인:'):
                    self.domain = line.replace('도메인:', '').strip()
                elif line.startswith('키워드:'):
                    keywords_str = line.replace('키워드:', '').strip()
                    self.keywords = [k.strip() for k in keywords_str.split(',')]
                elif line.startswith('전문용어:'):
                    terms_str = line.replace('전문용어:', '').strip()
                    self.technical_terms = [t.strip() for t in terms_str.split(',')]

            self.is_ready = True
            print(f"[Metadata] 추출 완료")
            print(f"   도메인: {self.domain}")
            print(f"   키워드: {', '.join(self.keywords[:5])}...")
            print(f"   전문용어: {', '.join(self.technical_terms[:5])}...")

        except Exception as e:
            print(f"[Metadata] 추출 실패: {e}")
            self.is_ready = False

    def get_metadata_context(self):
        """
        질의 교정에 사용할 메타데이터 컨텍스트를 반환합니다.
        """
        if not self.is_ready:
            return ""

        context = f"""[문서 메타데이터]
도메인: {self.domain}
주요 키워드: {', '.join(self.keywords)}
전문 용어: {', '.join(self.technical_terms)}"""
        return context

# ==========================================
# 4. 검색 엔진 (Retrievers)
# ==========================================
class VectorStore:
    def __init__(self):
        self.client = chromadb.Client() # 메모리 모드 (휘발성)
        # Ollama 임베딩 함수 사용 (chromadb 기본 지원 아님, 커스텀 필요)
        # 여기서는 편의상 SentenceTransformer 대신 간단히 구현하거나
        # Ollama API를 직접 호출해서 임베딩을 가져와야 함.
        # 편의를 위해 chromadb의 DefaultEmbeddingFunction을 쓰지 않고 직접 주입 방식 사용.
        self.collection = self.client.create_collection(name="docs")

    def add_documents(self, chunks):
        ids = [str(i) for i in range(len(chunks))]
        # Ollama를 통해 임베딩 생성
        embeddings = []
        print("   [Vector] 임베딩 생성 중... (시간이 걸릴 수 있습니다)")
        client = get_ollama_client()  # 원격 호스트 지원
        for chunk in chunks:
            try:
                res = client.embeddings(model=Config.MODEL_EMBED, prompt=chunk)
                embeddings.append(res['embedding'])
            except Exception as e:
                print(f"   [Error] 임베딩 생성 실패: {e}")
                embeddings.append([0.0]*768) # 더미(실패 시)

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        print(f"   [Vector] {len(chunks)}개 청크 벡터 DB 저장 완료.")

    def search(self, query, top_k=5):
        # 쿼리 임베딩
        client = get_ollama_client()  # 원격 호스트 지원
        res = client.embeddings(model=Config.MODEL_EMBED, prompt=query)
        query_embedding = res['embedding']
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        # ChromaDB 결과 포맷 정리
        return results['documents'][0] if results['documents'] else []

class KeywordStore:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def add_documents(self, chunks):
        self.chunks = chunks
        # 간단한 공백 단위 토크나이징 (한국어 형태소 분석기 없이 약식 구현)
        tokenized_corpus = [doc.split(" ") for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"   [Keyword] BM25 인덱싱 완료.")

    def search(self, query, top_k=5):
        tokenized_query = query.split(" ")
        # 점수 계산
        doc_scores = self.bm25.get_scores(tokenized_query)
        # 상위 k개 인덱스 추출
        top_n_indexes = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in top_n_indexes]

# ==========================================
# 5. Reranker & Query Refiner (LLM 기반)
# ==========================================
def correct_query_with_metadata(query, metadata_store):
    """
    메타데이터를 사용하여 사용자 질문의 오타와 띄어쓰기를 교정하고,
    질문 유형도 함께 분석합니다.

    Returns:
        tuple: (corrected_query, query_type)
    """
    if not metadata_store.is_ready:
        # 메타데이터가 없으면 기본 교정
        return correct_query_basic(query)

    metadata_context = metadata_store.get_metadata_context()

    prompt = f"""당신은 질문 분석 전문가입니다. 다음 작업을 수행하세요:

1. [질문]의 오타와 띄어쓰기를 교정
2. [질문]의 유형을 분류

{metadata_context}

위 메타데이터의 전문 용어나 키워드를 우선 사용하여 교정하세요.

질문 유형 분류 기준:
- SEARCH: 특정 정보를 찾는 질문 (예: "재택근무 승인은?", "XG-200 출력은?", "가격은?", "주의할 점은?")
- SUMMARY: 문서 전체/전반적인 내용 요약 요청 (예: "문서 요약해줘", "전체 내용 정리", "이 문서가 뭔 내용이야?")
- COMPARE: 비교/대조 요청 (예: "A와 B 차이점", "장단점 비교")
- LIST: 모든 항목 나열 요청 (예: "모든 제품 목록", "전체 종류", "모든 기능")

[질문]
{query}

다음 형식으로만 출력하세요:
교정된질문: [교정된 질문]
유형: [SEARCH/SUMMARY/COMPARE/LIST]

예시:
교정된질문: 재택근무 승인 절차는?
유형: SEARCH
"""
    try:
        client = get_ollama_client()
        res = client.chat(model=Config.MODEL_CHAT, messages=[{'role': 'user', 'content': prompt}])
        content = res['message']['content'].strip()

        # 파싱
        corrected = query
        query_type = QueryType.SEARCH  # 기본값

        for line in content.split('\n'):
            if line.startswith('교정된질문:'):
                corrected = line.replace('교정된질문:', '').strip()
                corrected = corrected.replace('"', '').replace("'", "")
            elif line.startswith('유형:'):
                type_str = line.replace('유형:', '').strip().upper()
                if type_str in [QueryType.SEARCH, QueryType.SUMMARY, QueryType.COMPARE, QueryType.LIST]:
                    query_type = type_str
                # 혹시 유형이 문장에 포함되어 있을 경우
                elif QueryType.SUMMARY in type_str:
                    query_type = QueryType.SUMMARY
                elif QueryType.COMPARE in type_str:
                    query_type = QueryType.COMPARE
                elif QueryType.LIST in type_str:
                    query_type = QueryType.LIST

        return corrected, query_type
    except:
        return query, QueryType.SEARCH

def correct_query_basic(query):
    """
    기본 질의 교정 (메타데이터 없이)

    Returns:
        tuple: (corrected_query, query_type)
    """
    prompt = f"""당신은 질문 분석 전문가입니다. 다음 작업을 수행하세요:

1. [질문]의 오타와 띄어쓰기를 교정
2. [질문]의 유형을 분류

질문 유형 분류 기준:
- SEARCH: 특정 정보를 찾는 질문 (예: "재택근무 승인은?", "XG-200 출력은?", "가격은?", "주의할 점은?")
- SUMMARY: 문서 전체/전반적인 내용 요약 요청 (예: "문서 요약해줘", "전체 내용 정리", "이 문서가 뭔 내용이야?")
- COMPARE: 비교/대조 요청 (예: "A와 B 차이점", "장단점 비교")
- LIST: 모든 항목 나열 요청 (예: "모든 제품 목록", "전체 종류", "모든 기능")

[질문]
{query}

다음 형식으로만 출력하세요:
교정된질문: [교정된 질문]
유형: [SEARCH/SUMMARY/COMPARE/LIST]
"""
    try:
        client = get_ollama_client()
        res = client.chat(model=Config.MODEL_CHAT, messages=[{'role': 'user', 'content': prompt}])
        content = res['message']['content'].strip()

        # 파싱
        corrected = query
        query_type = QueryType.SEARCH

        for line in content.split('\n'):
            if line.startswith('교정된질문:'):
                corrected = line.replace('교정된질문:', '').strip()
                corrected = corrected.replace('"', '').replace("'", "")
            elif line.startswith('유형:'):
                type_str = line.replace('유형:', '').strip().upper()
                if type_str in [QueryType.SEARCH, QueryType.SUMMARY, QueryType.COMPARE, QueryType.LIST]:
                    query_type = type_str

        return corrected, query_type
    except:
        return query, QueryType.SEARCH

def rerank_documents(query, docs):
    """
    LLM을 사용하여 문서의 연관성을 평가하고 재정렬합니다.
    """
    print("   [Rerank] 문서 재순위 지정 중(LLM)...")
    scored_docs = []
    client = get_ollama_client()  # 원격 호스트 지원

    for doc in docs:
        prompt = f"""
당신은 검색 품질 관리자입니다.
아래의 [문서]가 [사용자 질문]에 답변하는 데 얼마나 유용한지 평가하세요.

[사용자 질문]
{query}

[문서]
{doc[:500]}...

이 문서가 질문과 관련이 있다면 'Yes', 전혀 관련이 없다면 'No'라고만 답하세요.
설명은 하지 마세요.
"""
        try:
            res = client.chat(model=Config.MODEL_RERANK, messages=[{'role': 'user', 'content': prompt}])
            content = res['message']['content'].strip().lower()
            score = 1 if 'yes' in content else 0
            if score == 1:
                scored_docs.append(doc)
        except:
            pass
            
    return scored_docs if scored_docs else docs[:3] # 실패 시 기본 반환

# ==========================================
# 5. Query Router (질문 유형 분류)
# ==========================================

# 질문 유형 상수
class QueryType:
    SEARCH = "SEARCH"       # 특정 정보 검색
    SUMMARY = "SUMMARY"     # 전체 요약
    COMPARE = "COMPARE"     # 비교/대조
    LIST = "LIST"           # 목록/나열

# 키워드 기반 빠른 분류 (더 엄격하게 조정)
QUERY_PATTERNS = {
    QueryType.SUMMARY: [
        # 명확한 요약/개요 요청만
        "요약해", "요약해줘", "정리해", "정리해줘", "개요", "전체 내용", "전반적",
        "summarize", "summary", "overview", "전체적으로",
        "뭔 내용", "무슨 내용", "내용이 뭐", "문서 전체",
    ],
    QueryType.LIST: [
        "모든", "전부", "목록", "리스트", "나열해", "종류 전부", "몇 가지",
        "all", "list", "every", "모두", "전체 종류",
    ],
    QueryType.COMPARE: [
        "비교", "차이", "vs", "versus", "장단점", "다른점", "공통점",
        "compare", "difference", "pros and cons",
    ],
}

def classify_query_fast(query):
    """
    키워드 기반 빠른 질문 유형 분류 (LLM 호출 없음)
    """
    query_lower = query.lower()

    for query_type, keywords in QUERY_PATTERNS.items():
        if any(kw in query_lower for kw in keywords):
            return query_type

    return QueryType.SEARCH  # 기본값

def classify_query_llm(query):
    """
    LLM 기반 정확한 질문 유형 분류
    """
    prompt = f"""다음 질문의 유형을 분류하세요.

질문: {query}

유형:
- SEARCH: 특정 정보를 찾는 질문 (예: "XG-200 출력은?", "A의 가격은?", "B가 뭐야?")
- SUMMARY: 전체 요약, 개요, 매뉴얼/가이드 요청 (예: "문서 요약해줘", "전체 절차 알려줘", "매뉴얼 알려줘", "어떻게 하는지 알려줘")
- COMPARE: 비교/대조 요청 (예: "A와 B 차이점", "장단점 비교해줘")
- LIST: 목록/나열 요청 (예: "모든 제품 목록", "기능 전부 알려줘", "종류가 뭐가 있어")

유형 하나만 출력하세요 (SEARCH/SUMMARY/COMPARE/LIST):"""

    try:
        client = get_ollama_client()
        res = client.chat(model=Config.MODEL_CHAT, messages=[{'role': 'user', 'content': prompt}])
        result = res['message']['content'].strip().upper()

        # 유효한 유형인지 확인
        if result in [QueryType.SEARCH, QueryType.SUMMARY, QueryType.COMPARE, QueryType.LIST]:
            return result
        # 결과에서 유형 추출 시도
        for qt in [QueryType.SUMMARY, QueryType.LIST, QueryType.COMPARE, QueryType.SEARCH]:
            if qt in result:
                return qt
        return QueryType.SEARCH
    except:
        return classify_query_fast(query)  # 실패 시 키워드 기반으로 폴백

def hierarchical_summary(full_doc, chunk_size=None):
    """
    문서가 너무 길 때 계층적 요약을 수행합니다.
    1단계: 각 청크를 요약
    2단계: 요약들을 합쳐서 최종 컨텍스트 생성
    """
    if chunk_size is None:
        chunk_size = Config.SUMMARY_CHUNK_SIZE

    # 문서를 큰 청크로 분할 (오버랩 없이)
    chunks = chunk_text(full_doc, chunk_size, overlap=0)

    print(f"   [Summary] 계층적 요약 시작: {len(chunks)}개 섹션")

    client = get_ollama_client()
    summaries = []

    for i, chunk in enumerate(chunks):
        print(f"   [Summary] 섹션 {i+1}/{len(chunks)} 요약 중...")
        prompt = f"""다음 내용을 3-5문장으로 핵심만 요약하세요.
중요한 수치, 이름, 용어는 반드시 포함하세요.

내용:
{chunk}

요약:"""

        try:
            res = client.chat(model=Config.MODEL_CHAT, messages=[{'role': 'user', 'content': prompt}])
            summary = res['message']['content'].strip()
            summaries.append(f"[섹션 {i+1}]\n{summary}")
        except Exception as e:
            print(f"   [Warning] 섹션 {i+1} 요약 실패: {e}")
            # 실패 시 원본 청크의 앞부분 사용
            summaries.append(f"[섹션 {i+1}]\n{chunk[:500]}...")

    combined = "\n\n".join(summaries)

    # 결합된 요약이 여전히 너무 크면 재귀적으로 다시 요약
    max_context = int(Config.NUM_CTX * Config.MAX_CONTEXT_RATIO * 4)  # 대략 글자 수 추정
    if len(combined) > max_context:
        print(f"   [Summary] 요약 결과가 여전히 큼. 2차 요약 수행...")
        return hierarchical_summary(combined, chunk_size)

    print(f"   [Summary] 계층적 요약 완료 ({len(full_doc)}자 → {len(combined)}자)")
    return combined

def extract_comparison_entities(query):
    """
    비교 질문에서 비교 대상 엔티티를 추출합니다.
    """
    prompt = f"""다음 질문에서 비교 대상이 되는 항목들을 추출하세요.

질문: {query}

쉼표로 구분하여 항목만 출력하세요 (예: A, B, C):"""

    try:
        client = get_ollama_client()
        res = client.chat(model=Config.MODEL_CHAT, messages=[{'role': 'user', 'content': prompt}])
        entities = [e.strip() for e in res['message']['content'].split(',')]
        return entities if entities else [query]
    except:
        return [query]

class SummaryCache:
    """
    사전 생성된 요약을 저장하는 캐시 클래스
    인덱싱 단계에서 미리 요약을 생성해두고 질문 시 바로 사용
    """
    def __init__(self):
        self.full_summary = None      # 전체 문서 요약
        self.section_summaries = []   # 섹션별 요약 리스트
        self.is_ready = False

    def generate(self, full_doc):
        """
        문서 로드 시 사전 요약을 생성합니다.
        """
        print("\n[Pre-Summary] 사전 요약 생성 중... (인덱싱 단계)")

        max_doc_size = int(Config.NUM_CTX * Config.MAX_CONTEXT_RATIO * 4)

        # 문서가 작으면 전체 문서를 그대로 사용
        if len(full_doc) <= max_doc_size:
            self.full_summary = full_doc
            print(f"[Pre-Summary] 문서가 작음({len(full_doc)}자). 전체 문서를 캐시합니다.")
            self.is_ready = True
            return

        # 문서가 크면 계층적 요약 수행
        print(f"[Pre-Summary] 문서가 큼({len(full_doc)}자). 계층적 요약을 수행합니다...")

        chunk_size = Config.SUMMARY_CHUNK_SIZE
        chunks = chunk_text(full_doc, chunk_size, overlap=0)

        print(f"[Pre-Summary] {len(chunks)}개 섹션으로 분할")

        client = get_ollama_client()
        summaries = []

        for i, chunk in enumerate(chunks):
            print(f"   [Pre-Summary] 섹션 {i+1}/{len(chunks)} 요약 중...")
            prompt = f"""다음 내용을 3-5문장으로 핵심만 요약하세요.
중요한 수치, 이름, 용어는 반드시 포함하세요.

내용:
{chunk}

요약:"""

            try:
                res = client.chat(model=Config.MODEL_CHAT, messages=[{'role': 'user', 'content': prompt}])
                summary = res['message']['content'].strip()
                summaries.append(f"[섹션 {i+1}]\n{summary}")
            except Exception as e:
                print(f"   [Warning] 섹션 {i+1} 요약 실패: {e}")
                summaries.append(f"[섹션 {i+1}]\n{chunk[:500]}...")

        self.section_summaries = summaries
        combined = "\n\n".join(summaries)

        # 결합된 요약이 여전히 크면 2차 요약
        if len(combined) > max_doc_size:
            print(f"[Pre-Summary] 1차 요약 결과가 큼({len(combined)}자). 2차 요약 수행...")
            prompt = f"""다음은 문서의 섹션별 요약입니다. 이를 하나의 통합된 요약으로 정리하세요.
핵심 내용, 주요 절차, 중요한 수치를 포함하세요.

{combined}

통합 요약:"""
            try:
                res = client.chat(model=Config.MODEL_CHAT, messages=[{'role': 'user', 'content': prompt}])
                self.full_summary = res['message']['content'].strip()
            except:
                self.full_summary = combined
        else:
            self.full_summary = combined

        print(f"[Pre-Summary] 사전 요약 완료 ({len(full_doc)}자 → {len(self.full_summary)}자)")
        self.is_ready = True

    def get_summary(self):
        """캐시된 요약을 반환합니다."""
        if self.is_ready and self.full_summary:
            return self.full_summary
        return None

# ==========================================
# 6. 메인 로직
# ==========================================
def main():
    print("=== Advanced Multi-Mode RAG Bot ===")
    
    # 1. 문서 로드
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"파일 경로(인자): {file_path}")
    else:
        file_path = input(f"분석할 파일 경로 입력 (기본값: {Config.DOC_PATH}): ").strip() or Config.DOC_PATH
    
    text = load_document(file_path)
    if not text: return

    print(f"\n[System] 문서 로드 성공 ({len(text)}자). 청크 생성 중...")
    chunks = chunk_text(text, Config.CHUNK_SIZE, Config.CHUNK_OVERLAP)
    print(f"[System] 총 {len(chunks)}개의 청크 생성됨.")

    # 2. 메타데이터 추출
    metadata_store = MetadataStore()
    metadata_store.extract_metadata(text)

    # 3. 인덱싱 (초기화)
    vector_store = VectorStore()
    keyword_store = KeywordStore()

    # 두 스토어 모두에 데이터 주입
    vector_store.add_documents(chunks)
    keyword_store.add_documents(chunks)

    # 4. 사전 요약 생성 (선택적)
    summary_cache = SummaryCache()
    if Config.PRE_SUMMARIZE:
        summary_cache.generate(text)
    else:
        print("[Pre-Summary] 사전 요약 비활성화됨 (Config.PRE_SUMMARIZE=False)")

    while True:
        print("\n" + "="*40)
        print(" 1. 파일 전체 검색 (No-RAG, Context Stuffing)")
        print(" 2. 벡터 검색 (Vector Store)")
        print(" 3. 키워드 검색 (BM25)")
        print(" 4. 하이브리드 검색 (Hybrid + Rerank)")
        print(" 5. 자동 모드 (Query Router) ★ 추천")
        print(" q. 종료")
        print("="*40)

        mode = input("검색 모드를 선택하세요 (1-5): ").strip()
        if mode.lower() in ['q', 'exit', 'ㅂ']: break
        
        original_query = input("\n질문: ").strip()
        if not original_query: continue

        # 메타데이터 기반 질문 교정 및 유형 분석 (벡터 검색 없이)
        print("   [Query] 질문 분석 중... (교정 + 유형 판단)")
        query, detected_query_type = correct_query_with_metadata(original_query, metadata_store)

        if query != original_query:
            print(f"   => 교정된 질문: {query}")
        else:
            print(f"   => 질문: {query}")
        print(f"   => 질문 유형: {detected_query_type}")

        context = ""
        
        # --- 검색 단계 ---
        if mode == '1':
            context = text # 전체 텍스트
            print(f"[Mode 1] 전체 문서({len(text)}자)를 컨텍스트로 사용합니다.")
            
        elif mode == '2':
            docs = vector_store.search(query, top_k=Config.TOP_K)
            context = "\n---\n".join(docs)
            print(f"[Mode 2] 벡터 유사도 상위 {len(docs)}개 청크 사용.")
            
        elif mode == '3':
            docs = keyword_store.search(query, top_k=Config.TOP_K)
            context = "\n---\n".join(docs)
            print(f"[Mode 3] 키워드 매칭 상위 {len(docs)}개 청크 사용.")
            
        elif mode == '4':
            vec_docs = vector_store.search(query, top_k=Config.TOP_K)
            key_docs = keyword_store.search(query, top_k=Config.TOP_K)

            # 중복 제거해서 합치기
            combined_docs = list(set(vec_docs + key_docs))
            print(f"[Hybrid] 1차 검색 완료 ({len(combined_docs)}개 문서). Reranking 시작...")

            # 리랭크
            final_docs = rerank_documents(query, combined_docs)
            context = "\n---\n".join(final_docs)
            print(f"[Mode 4] 최종 선정된 {len(final_docs)}개 청크 사용.")

        elif mode == '5':
            # === 자동 모드 (Query Router) ===
            # 이미 질의 교정 단계에서 LLM이 유형을 판단했으므로 그것을 사용
            query_type = detected_query_type
            print(f"   [Router] LLM이 분석한 질문 유형 사용: {query_type}")

            # 2. 유형별 처리
            if query_type == QueryType.SUMMARY:
                # 요약형: 사전 생성된 요약 사용 (없으면 실시간 생성)
                cached = summary_cache.get_summary()
                if cached:
                    context = cached
                    print(f"[Mode 5-SUMMARY] 사전 생성된 요약 사용 ({len(cached)}자) ⚡ 빠른 응답")
                else:
                    # 사전 요약이 없으면 기존 방식으로 실시간 생성
                    max_doc_size = int(Config.NUM_CTX * Config.MAX_CONTEXT_RATIO * 4)
                    if len(text) <= max_doc_size:
                        context = text
                        print(f"[Mode 5-SUMMARY] 전체 문서({len(text)}자)를 컨텍스트로 사용합니다.")
                    else:
                        print(f"[Mode 5-SUMMARY] 문서가 큼({len(text)}자). 계층적 요약을 수행합니다.")
                        context = hierarchical_summary(text)

            elif query_type == QueryType.LIST:
                # 목록형: 더 많은 청크 검색
                extended_top_k = min(Config.TOP_K * 4, len(chunks))
                vec_docs = vector_store.search(query, top_k=extended_top_k)
                key_docs = keyword_store.search(query, top_k=extended_top_k)
                combined_docs = list(set(vec_docs + key_docs))
                context = "\n---\n".join(combined_docs)
                print(f"[Mode 5-LIST] 확장 검색으로 {len(combined_docs)}개 청크 사용.")

            elif query_type == QueryType.COMPARE:
                # 비교형: 각 엔티티별로 검색 후 병합
                entities = extract_comparison_entities(query)
                print(f"   [Router] 비교 대상: {entities}")

                all_docs = []
                for entity in entities:
                    vec_docs = vector_store.search(entity, top_k=Config.TOP_K)
                    key_docs = keyword_store.search(entity, top_k=Config.TOP_K)
                    all_docs.extend(vec_docs + key_docs)

                combined_docs = list(set(all_docs))
                final_docs = rerank_documents(query, combined_docs)
                context = "\n---\n".join(final_docs)
                print(f"[Mode 5-COMPARE] {len(entities)}개 항목 검색, {len(final_docs)}개 청크 사용.")

            else:  # QueryType.SEARCH
                # 검색형: 하이브리드 검색 + 리랭킹 (기존 모드 4와 동일)
                vec_docs = vector_store.search(query, top_k=Config.TOP_K)
                key_docs = keyword_store.search(query, top_k=Config.TOP_K)
                combined_docs = list(set(vec_docs + key_docs))
                final_docs = rerank_documents(query, combined_docs)
                context = "\n---\n".join(final_docs)
                print(f"[Mode 5-SEARCH] 하이브리드 검색, {len(final_docs)}개 청크 사용.")

        else:
            print("잘못된 입력입니다.")
            continue

        # --- 답변 생성 단계 ---
        print("\n[AI 답변 생성 중...]\n")
        prompt = f"""
당신은 유능한 AI 어시스턴트입니다. 아래의 [배경 지식]을 참고하여 사용자의 [질문]에 답변하세요.
만약 배경 지식에 정답이 없다면, 솔직하게 모른다고 말하세요.
반드시 사용자가 질문한 언어와 동일한 언어로 답변하세요 (예: 한국어 질문 -> 한국어 답변).

[배경 지식]
{context}

[질문]
{query}
"""
        try:
            client = ollama.Client(host=Config.OLLAMA_HOST)
            stream = client.chat(
                model=Config.MODEL_CHAT,
                messages=[{'role': 'user', 'content': prompt}],
                stream=True,
                options={'num_ctx': Config.NUM_CTX}
            )
            for chunk in stream:
                print(chunk['message']['content'], end="", flush=True)
            print()
        except Exception as e:
            print(f"Ollama 오류: {e}")

if __name__ == "__main__":
    main()
