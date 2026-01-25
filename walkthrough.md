# No-RAG & Hybrid RAG Bot 프로젝트 완료

이 프로젝트는 로컬 LLM(Ollama)을 활용한 두 가지 버전의 Q&A 봇을 포함합니다.

## 1. 프로젝트 파일 구성

| 파일명 | 설명 | 비고 |
| :-- | :-- | :-- |
| `no_rag_bot.py` | **Context Stuffing 방식**. 문서 전체를 한 번에 넣고 질문. | 작은 문서에 최적. 가장 정확함. |
| `advanced_rag_bot.py` | **Hybrid RAG 방식**. 벡터+키워드 검색 후 리랭킹. | 대용량 문서에 최적. 4가지 모드 지원. **심층 텍스트 추출 & 문맥 인식 교정 탑재** |
| `requirements.txt` | 필요 라이브러리 목록 파일. | `pip install -r requirements.txt` |
| `data.docx` / `test.md` | 테스트용 샘플 데이터. | |

## 2. 주요 업데이트 기능
- **Deep XML Extraction**: `.docx` 파일의 텍스트 상자, 도형, 표 깊은 곳에 숨겨진 텍스트까지 모두 추출합니다. (기존 라이브러리 한계 극복)
- **Context-Aware Query Correction**: 사용자가 질문을 입력하면, 1차적으로 벡터 검색을 수행하여 문서 내 용어를 파악한 뒤, LLM이 오타와 용어를 교정합니다. (예: "출자 규정" -> "출장 규정")

## 3. 설치 및 실행

```bash
# 가상환경 사용 권장
pip install -r requirements.txt

# 실행 (파일 경로 인자 지원)
python advanced_rag_bot.py "d:\Downloads\MyDoc.docx"
```

## 4. 모드 선택 가이드
- **모드 1 (No-RAG)**: 문서가 작을 때 (가장 정확)
- **모드 4 (Hybrid)**: 문서가 클 때 (벡터+키워드+리랭크+질문교정의 강력한 조합)
