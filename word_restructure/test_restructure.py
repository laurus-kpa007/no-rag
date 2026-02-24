"""
테스트용 더미 문서 생성 및 파싱 테스트
Ollama 없이도 문서 추출/파싱 로직을 테스트할 수 있습니다.

사용법:
    python -m word_restructure.test_restructure
"""

import os
import sys
from docx import Document


def create_test_document_structured(path: str):
    """헤딩이 있는 구조화된 테스트 문서 생성"""
    doc = Document()

    doc.add_heading('인공지능 기술 보고서', 0)

    doc.add_heading('1. 서론', level=1)
    doc.add_paragraph(
        '인공지능(AI)은 현대 기술의 핵심 분야로 자리잡았습니다. '
        '본 보고서에서는 AI의 현재 동향과 미래 전망을 분석합니다.'
    )
    doc.add_paragraph(
        '최근 대규모 언어 모델(LLM)의 발전으로 자연어 처리 분야에서 '
        '혁명적인 변화가 일어나고 있습니다.'
    )

    doc.add_heading('2. 기술 현황', level=1)

    doc.add_heading('2.1 대규모 언어 모델', level=2)
    doc.add_paragraph(
        'GPT, Claude, Gemma 등 다양한 LLM이 등장하여 텍스트 생성, '
        '요약, 번역 등 다양한 작업을 수행하고 있습니다.'
    )
    doc.add_paragraph('주요 LLM 모델:', style='List Bullet')
    doc.add_paragraph('GPT-4: OpenAI의 대표 모델', style='List Bullet')
    doc.add_paragraph('Claude: Anthropic의 안전한 AI', style='List Bullet')
    doc.add_paragraph('Gemma: Google의 오픈소스 모델', style='List Bullet')

    doc.add_heading('2.2 컴퓨터 비전', level=2)
    doc.add_paragraph(
        '이미지 인식, 객체 탐지, 자율주행 등 컴퓨터 비전 분야에서도 '
        'AI 기술이 빠르게 발전하고 있습니다.'
    )

    doc.add_heading('3. 산업 적용 사례', level=1)

    # 표 추가
    table = doc.add_table(rows=4, cols=3)
    table.style = 'Table Grid'
    headers = ['분야', '적용 사례', '효과']
    for j, header in enumerate(headers):
        table.cell(0, j).text = header

    data = [
        ['의료', '질병 진단 보조', '진단 정확도 30% 향상'],
        ['금융', '사기 탐지', '사기 적발률 50% 향상'],
        ['제조', '품질 검사 자동화', '불량률 20% 감소'],
    ]
    for i, row_data in enumerate(data):
        for j, cell_text in enumerate(row_data):
            table.cell(i + 1, j).text = cell_text

    doc.add_heading('4. 향후 전망', level=1)
    doc.add_paragraph(
        'AI 기술은 앞으로 더욱 발전하여 인간의 삶을 크게 변화시킬 것으로 예상됩니다. '
        '특히 멀티모달 AI, 에이전트 AI, 소형화된 온디바이스 AI가 주요 트렌드가 될 것입니다.'
    )

    doc.add_heading('5. 결론', level=1)
    doc.add_paragraph(
        'AI 기술의 발전은 기회와 도전을 동시에 가져옵니다. '
        '기술의 책임있는 개발과 활용이 중요합니다.'
    )

    doc.save(path)
    print(f"[TEST] 구조화된 테스트 문서 생성: {path}")


def create_test_document_unstructured(path: str):
    """헤딩 없는 비정형 테스트 문서 생성"""
    doc = Document()

    doc.add_paragraph('회의록')
    doc.add_paragraph('')
    doc.add_paragraph('일시: 2026년 2월 24일')
    doc.add_paragraph('장소: 본사 회의실')
    doc.add_paragraph('참석자: 김철수, 이영희, 박민수, 정수진')
    doc.add_paragraph('')
    doc.add_paragraph(
        '첫 번째 안건으로 신규 프로젝트 일정에 대해 논의했습니다. '
        '김철수 팀장이 전체 일정을 설명했고, 3월 말까지 1차 프로토타입을 '
        '완성하기로 합의했습니다.'
    )
    doc.add_paragraph(
        '두 번째 안건으로 예산 배분에 대해 논의했습니다. '
        '인프라 구축에 40%, 인력 비용에 35%, 기타 운영비에 25%를 '
        '배분하기로 했습니다.'
    )
    doc.add_paragraph(
        '세 번째 안건으로 인력 충원 계획을 논의했습니다. '
        'AI 엔지니어 2명, 프론트엔드 개발자 1명을 3월 중으로 '
        '채용하기로 결정했습니다.'
    )
    doc.add_paragraph('')
    doc.add_paragraph('다음 회의 일정: 2026년 3월 3일')

    doc.save(path)
    print(f"[TEST] 비정형 테스트 문서 생성: {path}")


def test_extractor():
    """문서 추출 로직 테스트"""
    from .extractor import extract_document, elements_to_sections, sections_to_text_summary

    test_dir = 'test_data'
    os.makedirs(test_dir, exist_ok=True)

    # 테스트 문서 생성
    structured_path = os.path.join(test_dir, 'test_structured.docx')
    unstructured_path = os.path.join(test_dir, 'test_unstructured.docx')

    create_test_document_structured(structured_path)
    create_test_document_unstructured(unstructured_path)

    # 구조화된 문서 테스트
    print("\n" + "=" * 50)
    print("구조화된 문서 추출 테스트")
    print("=" * 50)
    extracted = extract_document(structured_path)
    print(f"제목: {extracted.title}")
    print(f"요소 수: {len(extracted.elements)}")
    print(f"텍스트 길이: {len(extracted.raw_text)}자")

    for elem in extracted.elements:
        prefix = f"[{elem.type}]"
        if elem.type == 'heading':
            prefix += f"(L{elem.level})"
        content_preview = elem.content[:60] + "..." if len(elem.content) > 60 else elem.content
        print(f"  {prefix} {content_preview}")

    sections = elements_to_sections(extracted.elements)
    print(f"\n섹션 수: {len(sections)}")
    print(sections_to_text_summary(sections))

    # 비정형 문서 테스트
    print("\n" + "=" * 50)
    print("비정형 문서 추출 테스트")
    print("=" * 50)
    extracted2 = extract_document(unstructured_path)
    print(f"제목: {extracted2.title}")
    print(f"요소 수: {len(extracted2.elements)}")

    sections2 = elements_to_sections(extracted2.elements)
    print(f"섹션 수: {len(sections2)}")

    print("\n[TEST] 추출 테스트 완료!")
    return structured_path


def main():
    """테스트 메인"""
    print("=" * 55)
    print("  Word Document Restructuring Tool - 테스트")
    print("=" * 55)

    # Step 1: 추출 테스트 (Ollama 불필요)
    structured_path = test_extractor()

    # Step 2: 전체 파이프라인 테스트 (Ollama 필요)
    print("\n" + "=" * 55)
    print("전체 파이프라인 테스트 (Ollama 필요)")
    print("=" * 55)

    try:
        import ollama
        client = ollama.Client(host='http://localhost:11434')
        client.list()
        print("[TEST] Ollama 연결 성공")
    except Exception as e:
        print(f"[TEST] Ollama 연결 실패: {e}")
        print("[TEST] 전체 파이프라인 테스트를 건너뜁니다.")
        print("[TEST] Ollama를 시작하고 gemma3:27b를 설치하세요:")
        print("       ollama pull gemma3:27b")
        return

    # 전체 실행
    print(f"\n[TEST] 전체 파이프라인 실행: {structured_path}")
    from .main import main as run_main
    sys.argv = ['test', structured_path, '-o', 'test_output']
    try:
        run_main()
        print("\n[TEST] 전체 파이프라인 테스트 성공!")
    except SystemExit:
        pass
    except Exception as e:
        print(f"\n[TEST] 전체 파이프라인 테스트 실패: {e}")


if __name__ == "__main__":
    main()
