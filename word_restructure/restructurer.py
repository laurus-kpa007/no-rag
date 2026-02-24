"""
콘텐츠 재구성 모듈
RestructurePlan에 따라 원본 요소들을 새로운 구조로 재배치합니다.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config, get_ollama_client, get_llm_options
from .extractor import DocumentElement, ExtractedDocument, elements_to_sections
from .analyzer import TocItem, RestructurePlan


@dataclass
class RestructuredSection:
    """재구성된 섹션"""
    level: int
    title: str
    content_elements: List[DocumentElement] = field(default_factory=list)
    subsections: List['RestructuredSection'] = field(default_factory=list)


@dataclass
class RestructuredDocument:
    """재구성된 문서"""
    title: str
    sections: List[RestructuredSection]
    document_type: str
    main_topic: str


def restructure_document(
    extracted: ExtractedDocument,
    plan: RestructurePlan,
    refine: bool = False,
) -> RestructuredDocument:
    """
    추출된 문서를 재구성 계획에 따라 재구성합니다.

    Args:
        extracted: 추출된 문서 객체
        plan: 재구성 계획
        refine: True이면 LLM으로 내용을 다듬기

    Returns:
        RestructuredDocument: 재구성된 문서
    """
    print("[재구성] 문서 재구성 시작...")

    # 원본 섹션 매핑
    original_sections = elements_to_sections(extracted.elements)

    # TOC 항목을 기반으로 재구성된 섹션 생성
    restructured_sections = _build_sections(plan.toc, original_sections, refine)

    print(f"[재구성] 완료: {_count_sections(restructured_sections)}개 섹션 생성")

    return RestructuredDocument(
        title=plan.title,
        sections=restructured_sections,
        document_type=plan.analysis.document_type,
        main_topic=plan.analysis.main_topic,
    )


def _build_sections(
    toc_items: List[TocItem],
    original_sections: list,
    refine: bool,
) -> List[RestructuredSection]:
    """TOC 항목을 재구성된 섹션으로 변환"""
    sections = []

    for item in toc_items:
        # 이 섹션에 매핑된 원본 요소 수집
        content_elements = _collect_elements(item.source_sections, original_sections)

        # 내용 다듬기 (선택)
        if refine and content_elements:
            content_elements = _refine_content(item.title, content_elements)

        # 하위 섹션 재귀 처리
        subsections = _build_sections(item.subsections, original_sections, refine)

        sections.append(RestructuredSection(
            level=item.level,
            title=item.title,
            content_elements=content_elements,
            subsections=subsections,
        ))

    return sections


def _collect_elements(
    source_indices: List[int],
    original_sections: list,
) -> List[DocumentElement]:
    """원본 섹션 인덱스에서 요소들을 수집"""
    elements = []
    for idx in source_indices:
        if 0 <= idx < len(original_sections):
            section = original_sections[idx]
            for elem in section['elements']:
                # heading은 새 구조에서 재생성되므로 내용만 가져옴
                if elem.type != 'heading':
                    elements.append(elem)
    return elements


def _refine_content(title: str, elements: List[DocumentElement]) -> List[DocumentElement]:
    """LLM으로 섹션 내용을 다듬기"""
    # 텍스트 요소만 다듬기 (표는 원본 유지)
    text_elements = [e for e in elements if e.type in ('paragraph', 'list_item')]
    non_text_elements = [e for e in elements if e.type not in ('paragraph', 'list_item')]

    if not text_elements:
        return elements

    original_text = "\n".join(e.content for e in text_elements)

    if len(original_text) < 50:
        return elements

    try:
        client = get_ollama_client()
        prompt = f"""아래는 "{title}" 섹션의 내용입니다.
내용의 의미를 보존하면서 더 자연스럽고 읽기 쉽게 다듬어주세요.
원본 정보를 누락하거나 새로운 정보를 추가하지 마세요.
다듬어진 텍스트만 출력하세요.

[원본]
{original_text}"""

        res = client.chat(
            model=Config.MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options=get_llm_options(Config.TEMPERATURE_REFINE),
        )
        refined_text = res['message']['content'].strip()

        # 다듬어진 텍스트를 단락으로 분할하여 새 요소 생성
        paragraphs = [p.strip() for p in refined_text.split('\n\n') if p.strip()]
        refined_elements = []
        for para in paragraphs:
            # 리스트 아이템 감지
            lines = para.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(('- ', '* ', '• ')) or (len(line) > 2 and line[0].isdigit() and line[1] in '.)'):
                    content = line.lstrip('-*• ').lstrip('0123456789.)').strip()
                    refined_elements.append(DocumentElement(
                        type='list_item', content=content, level=0
                    ))
                else:
                    refined_elements.append(DocumentElement(
                        type='paragraph', content=line, level=0
                    ))

        # 비텍스트 요소를 끝에 추가
        refined_elements.extend(non_text_elements)
        return refined_elements

    except Exception as e:
        print(f"  [다듬기] 실패 (원본 유지): {e}")
        return elements


def _count_sections(sections: List[RestructuredSection]) -> int:
    """총 섹션 수 카운트 (재귀)"""
    count = len(sections)
    for section in sections:
        count += _count_sections(section.subsections)
    return count
