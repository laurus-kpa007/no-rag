"""
Docling 기반 문서 재구조화 설정 모듈
Ollama LLM 및 Docling 파서 설정을 관리합니다.
"""

import os
from urllib.parse import urlparse

import ollama


class Config:
    # Ollama 서버 설정 (환경변수 OLLAMA_HOST로 오버라이드 가능)
    OLLAMA_HOST = os.environ.get('OLLAMA_HOST', 'http://70.30.171.45:11434')

    # 모델 설정
    MODEL = 'gemma3:27b'

    # LLM 하이퍼파라미터
    NUM_CTX = 32768
    TEMPERATURE = 0.3          # 구조 분석용 낮은 temperature
    TEMPERATURE_STRUCTURE = 0.4  # Pass 2 구조화용 (다양한 계층 생성 유도)
    TEMPERATURE_REFINE = 0.5   # 내용 다듬기용
    TOP_P = 0.9
    TOP_K = 40
    REPEAT_PENALTY = 1.1

    # 출력 설정
    OUTPUT_DIR = 'output'

    # 문서 처리 설정
    MAX_CONTENT_CHARS = 60000  # LLM에 전달할 최대 문자 수

    # 지원 파일 형식
    SUPPORTED_FORMATS = {'.docx', '.pdf', '.pptx', '.html', '.md'}


def get_ollama_client():
    """OLLAMA_HOST 설정을 사용하는 클라이언트 반환 (프록시 자동 우회)"""
    parsed = urlparse(Config.OLLAMA_HOST)
    target_host = parsed.hostname or 'localhost'

    no_proxy = os.environ.get('NO_PROXY', os.environ.get('no_proxy', ''))
    if target_host not in no_proxy:
        entries = [e.strip() for e in no_proxy.split(',') if e.strip()]
        entries.append(target_host)
        os.environ['NO_PROXY'] = ','.join(entries)

    return ollama.Client(host=Config.OLLAMA_HOST)


def get_llm_options(temperature=None):
    """LLM 호출 시 사용할 options 딕셔너리 반환"""
    return {
        'num_ctx': Config.NUM_CTX,
        'temperature': temperature or Config.TEMPERATURE,
        'top_p': Config.TOP_P,
        'top_k': Config.TOP_K,
        'repeat_penalty': Config.REPEAT_PENALTY,
    }
