"""
콘텐츠 재구성 모듈
RestructurePlan에 따라 원본 요소들을 새로운 구조로 재배치합니다.
"""

from dataclasses import dataclass, field
from typing import List, Set

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

    # 사용된 원본 섹션 추적 (중복 배치 방지)
    used_sections: Set[int] = set()

    # TOC 항목을 기반으로 재구성된 섹션 생성
    restructured_sections = _build_sections(
        plan.toc, original_sections, refine, used_sections
    )

    # 어디에도 배치되지 않은 원본 섹션 확인
    total_sections = len(original_sections)
    unused = [i for i in range(total_sections) if i not in used_sections]
    if unused:
        unused_elements = []
        for idx in unused:
            section = original_sections[idx]
            # 서두(level=0)이고 body가 비어있으면 건너뜀
            if section['level'] == 0 and not section['body_elements']:
                continue
            unused_elements.extend(section['body_elements'])

        if unused_elements:
            print(f"  [경고] {len(unused)}개 원본 섹션이 목차에 매핑되지 않아 '기타' 섹션에 추가")
            restructured_sections.append(RestructuredSection(
                level=1,
                title='기타',
                content_elements=unused_elements,
            ))

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
    used_sections: Set[int],
) -> List[RestructuredSection]:
    """TOC 항목을 재구성된 섹션으로 변환"""
    sections = []

    for item in toc_items:
        # 하위 섹션에서 사용할 인덱스를 미리 수집
        sub_source_indices = set()
        _collect_all_source_indices(item.subsections, sub_source_indices)

        # 이 섹션 자체에 매핑된 원본 요소 수집
        # (하위 섹션에서도 참조하는 인덱스는 하위에서 처리하므로 제외)
        own_indices = [
            idx for idx in item.source_sections
            if idx not in sub_source_indices
        ]
        content_elements = _collect_elements(own_indices, original_sections, used_sections)

        # 빈 섹션에 description 삽입 (매핑된 원본이 없거나 body가 비어있을 때)
        if not content_elements and (item.description or item.source_sections):
            fallback_text = item.description if item.description else f"({item.title} 섹션)"
            content_elements = [DocumentElement(
                type='paragraph', content=fallback_text, level=0
            )]

        # 진단 로그
        print(f"  [매핑] '{item.title}' ← 인덱스 {own_indices} → {len(content_elements)}개 요소"
              + (f" (하위: {len(item.subsections)}개)" if item.subsections else ""))

        # 내용 다듬기 (선택)
        if refine and content_elements:
            content_elements = _refine_content(item.title, content_elements)

        # 하위 섹션 재귀 처리
        subsections = _build_sections(item.subsections, original_sections, refine, used_sections)

        sections.append(RestructuredSection(
            level=item.level,
            title=item.title,
            content_elements=content_elements,
            subsections=subsections,
        ))

    return sections


def _collect_all_source_indices(toc_items: List[TocItem], indices: set):
    """TOC 항목 트리에서 모든 source_sections 인덱스를 수집"""
    for item in toc_items:
        indices.update(item.source_sections)
        _collect_all_source_indices(item.subsections, indices)


def _collect_elements(
    source_indices: List[int],
    original_sections: list,
    used_sections: Set[int],
) -> List[DocumentElement]:
    """원본 섹션 인덱스에서 본문 요소들을 수집 (중복 방지)"""
    elements = []
    for idx in source_indices:
        if idx in used_sections:
            continue  # 이미 다른 섹션에서 사용됨
        if 0 <= idx < len(original_sections):
            section = original_sections[idx]
            # body_elements 사용 (heading 제외된 본문만)
            elements.extend(section['body_elements'])
            used_sections.add(idx)
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
