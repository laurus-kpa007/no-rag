import os
import ollama


class Config:
    # Ollama 서버 설정
    OLLAMA_HOST = 'http://localhost:11434'

    # 모델 설정
    MODEL = 'gemma3:27b'

    # LLM 하이퍼파라미터
    NUM_CTX = 32768
    TEMPERATURE = 0.3          # 구조 분석용 낮은 temperature
    TEMPERATURE_REFINE = 0.5   # 내용 다듬기용
    TOP_P = 0.9
    TOP_K = 40
    REPEAT_PENALTY = 1.1

    # 출력 설정
    OUTPUT_DIR = 'output'

    # 문서 처리 설정
    MAX_CONTENT_CHARS = 60000  # LLM에 전달할 최대 문자 수 (컨텍스트 윈도우 고려)


def get_ollama_client():
    """OLLAMA_HOST 설정을 사용하는 클라이언트 반환"""
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
