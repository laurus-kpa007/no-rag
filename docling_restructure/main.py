"""
Docling Document Restructuring Tool - CLI 진입점
Docling으로 문서(DOCX, PDF 등)를 파싱하고 LLM으로 재구조화합니다.

사용법:
    python -m docling_restructure.main input.docx [-o output_dir] [--refine]
    python -m docling_restructure.main input.pdf [-o output_dir] [--refine]

옵션:
    input_file          입력 파일 경로 (.docx, .pdf 등)
    -o, --output-dir    출력 디렉토리 (기본값: output)
    --refine            LLM으로 내용 다듬기 (기본값: 구조만 재배치)
"""

import os
import sys
import argparse
import time

from .config import Config
from .parser import parse_document
from .analyzer import analyze_document, generate_restructure_plan, plan_to_text
from .restructurer import restructure_document
from .writer_docx import write_docx
from .writer_md import write_md


VERSION = '0.1.0'


def print_banner():
    print(f"""
{'='*60}
  Docling Document Restructuring Tool  v{VERSION}
{'='*60}
  LLM Model: {Config.MODEL}
  LLM Host:  {Config.OLLAMA_HOST}
  Context:   {Config.NUM_CTX:,} tokens
  Parser:    IBM Docling (AI-powered)
  Formats:   {', '.join(sorted(Config.SUPPORTED_FORMATS))}
{'='*60}
""")


def main():
    parser = argparse.ArgumentParser(
        description='Docling + LLM으로 문서를 구조화된 형식으로 재구성합니다.',
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        help='입력 파일 경로 (.docx, .pdf 등)',
    )
    parser.add_argument(
        '-o', '--output-dir',
        default=Config.OUTPUT_DIR,
        help=f'출력 디렉토리 (기본값: {Config.OUTPUT_DIR})',
    )
    parser.add_argument(
        '--refine',
        action='store_true',
        help='LLM으로 내용을 다듬습니다 (기본값: 구조만 재배치)',
    )

    args = parser.parse_args()

    print_banner()

    # 입력 파일 결정
    input_file = args.input_file
    if not input_file:
        input_file = input("입력 파일 경로를 입력하세요 (.docx, .pdf 등): ").strip()

    if not input_file:
        print("[오류] 입력 파일 경로가 필요합니다.")
        sys.exit(1)

    if not os.path.exists(input_file):
        print(f"[오류] 파일을 찾을 수 없습니다: {input_file}")
        sys.exit(1)

    ext = os.path.splitext(input_file)[1].lower()
    if ext not in Config.SUPPORTED_FORMATS:
        print(f"[오류] 지원하지 않는 파일 형식: {ext}")
        print(f"[INFO] 지원 형식: {', '.join(sorted(Config.SUPPORTED_FORMATS))}")
        sys.exit(1)

    start_time = time.time()

    # 출력 파일 경로 결정
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = args.output_dir
    output_docx = os.path.join(output_dir, f"{base_name}_restructured.docx")
    output_md = os.path.join(output_dir, f"{base_name}_restructured.md")

    print(f"[시작] 입력: {input_file} ({ext})")
    print(f"[시작] 출력: {output_docx}, {output_md}")
    if args.refine:
        print("[시작] 모드: 내용 다듬기 활성화")
    print()

    # Step 1: Docling 파싱
    print("=" * 45)
    print("Step 1/5: Docling 문서 파싱")
    print("=" * 45)
    try:
        parsed = parse_document(input_file)
        print(f"  → 섹션 {len(parsed.sections)}개 추출")
        print(f"  → 문서 제목: {parsed.title}")
        print(f"  → 구조 유형: {'구조화 문서' if parsed.is_structured else '비정형 문서'}")
        print(f"  → 원본 형식: {parsed.source_format}")
        print(f"  → 마크다운: {len(parsed.markdown_text)}자")
    except Exception as e:
        print(f"[오류] 문서 파싱 실패: {e}")
        print("[INFO] docling이 설치되어 있는지 확인하세요: pip install docling")
        sys.exit(1)

    # Step 2: 문서 분석 (Pass 1)
    print()
    print("=" * 45)
    print("Step 2/5: LLM 문서 분석")
    print("=" * 45)
    try:
        analysis = analyze_document(parsed)
        print(f"  → 문서 유형: {analysis.document_type}")
        print(f"  → 핵심 주제: {analysis.main_topic}")
        print(f"  → 주요 테마: {', '.join(analysis.key_themes)}")
        print(f"  → 콘텐츠 섹션: {len(analysis.content_sections)}개")
    except Exception as e:
        print(f"[오류] 문서 분석 실패: {e}")
        print(f"[INFO] Ollama 서버({Config.OLLAMA_HOST})가 실행 중인지 확인하세요.")
        print(f"[INFO]   ollama pull {Config.MODEL}")
        if 'proxy' in str(e).lower() or 'connect' in str(e).lower():
            print("[INFO] 프록시 문제가 의심됩니다. NO_PROXY 환경변수에 Ollama 호스트를 추가하세요.")
        sys.exit(1)

    # Step 3: 재구성 계획 생성 (Pass 2)
    print()
    print("=" * 45)
    print("Step 3/5: 재구성 계획 생성")
    print("=" * 45)
    try:
        plan = generate_restructure_plan(parsed, analysis)
        plan_text = plan_to_text(plan)
        print(plan_text)
    except Exception as e:
        print(f"[오류] 재구성 계획 생성 실패: {e}")
        sys.exit(1)

    # Step 4: 문서 재구성
    print()
    print("=" * 45)
    print("Step 4/5: 문서 재구성")
    print("=" * 45)
    try:
        restructured = restructure_document(parsed, plan, refine=args.refine)
    except Exception as e:
        print(f"[오류] 문서 재구성 실패: {e}")
        sys.exit(1)

    # Step 5: 출력 파일 생성
    print()
    print("=" * 45)
    print("Step 5/5: 출력 파일 생성")
    print("=" * 45)
    try:
        write_docx(restructured, output_docx)
        write_md(restructured, output_md)
    except Exception as e:
        print(f"[오류] 출력 파일 생성 실패: {e}")
        sys.exit(1)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"  완료! ({elapsed:.1f}초)")
    print(f"  출력 파일:")
    print(f"    - Word: {output_docx}")
    print(f"    - Markdown: {output_md}")
    print("=" * 60)


if __name__ == "__main__":
    main()
