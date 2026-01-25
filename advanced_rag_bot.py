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
    MODEL_CHAT = 'gemma3:4b'      # 답변 생성용 LLM
    MODEL_EMBED = 'nomic-embed-text' # 임베딩용 모델 (없으면 'ollama pull nomic-embed-text')
    MODEL_RERANK = 'gemma3:4b'    # 리랭킹용 LLM (가벼운 모델 권장)
    
    # 검색 설정
    CHUNK_SIZE = 500              # 청크 크기 (글자 수)
    CHUNK_OVERLAP = 50            # 청크 겹침 크기
    TOP_K = 5                     # 검색시 가져올 문서 수
    NUM_CTX = 32768               # LLM 컨텍스트 윈도우

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
# 3. 검색 엔진 (Retrievers)
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
        for chunk in chunks:
            try:
                # ollama.embeddings 사용
                res = ollama.embeddings(model=Config.MODEL_EMBED, prompt=chunk)
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
        res = ollama.embeddings(model=Config.MODEL_EMBED, prompt=query)
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
# 4. Reranker (LLM 기반)
# ==========================================
def rerank_documents(query, docs):
    """
    LLM을 사용하여 문서의 연관성을 평가하고 재정렬합니다.
    """
    print("   [Rerank] 문서 재순위 지정 중(LLM)...")
    scored_docs = []
    
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
            res = ollama.chat(model=Config.MODEL_RERANK, messages=[{'role': 'user', 'content': prompt}])
            content = res['message']['content'].strip().lower()
            score = 1 if 'yes' in content else 0
            if score == 1:
                scored_docs.append(doc)
        except:
            pass
            
    return scored_docs if scored_docs else docs[:3] # 실패 시 기본 반환

# ==========================================
# 5. 메인 로직
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

    # 2. 인덱싱 (초기화)
    vector_store = VectorStore()
    keyword_store = KeywordStore()
    
    # 두 스토어 모두에 데이터 주입
    vector_store.add_documents(chunks)
    keyword_store.add_documents(chunks)

    while True:
        print("\n" + "="*40)
        print(" 1. 파일 전체 검색 (No-RAG, Context Stuffing)")
        print(" 2. 벡터 검색 (Vector Store)")
        print(" 3. 키워드 검색 (BM25)")
        print(" 4. 하이브리드 검색 (Hybrid + Rerank)")
        print(" q. 종료")
        print("="*40)
        
        mode = input("검색 모드를 선택하세요 (1-4): ").strip()
        if mode.lower() in ['q', 'exit']: break
        
        query = input("\n질문: ").strip()
        if not query: continue

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
