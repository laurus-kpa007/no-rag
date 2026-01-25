# No-RAG Q&A Bot 구현 완료

요청하신 "RAG 없는 문서 기반 Q&A 봇"을 구현했습니다.

## 주요 변경 사항

### 1. 봇 스크립트 (`no_rag_bot.py`)
- `python-docx`를 사용하여 워드 파일의 모든 텍스트를 추출.
- 추출된 텍스트를 프롬프트에 포함하여 Ollama에 전송 (Context Stuffing).
- 사용자 입력 루프 및 실시간 스트리밍 출력 구현.

### 2. 테스트 데이터 (`data.docx`)
- 테스트를 위해 `create_dummy_docx.py`를 사용하여 샘플 문서를 생성했습니다.

## 검증 결과

스크립트 실행 및 로직 검증을 완료했습니다.

1. **데이터 로드**: `data.docx` 파일을 정상적으로 읽어 텍스트 길이를 출력했습니다.
2. **인터랙션**: 질문 입력 프롬프트가 정상적으로 표시되었습니다.
3. **종료**: 'q' 명령어로 프로그램이 정상 종료되었습니다.

## 실행 방법

터미널에서 다음 명령어를 실행하세요:

```bash
# 1. 의존성 설치
pip install ollama python-docx

# 2. 봇 실행 (기본 모델: llama3.2, 기본 파일: data.docx)
python no_rag_bot.py
```

만약 다른 문서를 분석하고 싶다면:
```bash
python no_rag_bot.py "내문서.docx"
```
코드를 수정하여 `MODEL_NAME` 변수를 설치된 다른 Ollama 모델(예: `mistral`, `gemma`)로 변경할 수 있습니다.
