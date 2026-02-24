"""
콘텐츠 재구성 모듈
RestructurePlan에 따라 Docling 파싱 결과를 새로운 구조로 재배치합니다.
"""

from dataclasses import dataclass, field
from typing import List, Set

from .config import Config, get_ollama_client, get_llm_options
from .parser import ParsedDocument, Section
from .analyzer import TocItem, RestructurePlan


@dataclass
class RestructuredSection:
    """재구성된 섹션"""
    level: int
    title: str
    content_md: str = ''                  # 마크다운 콘텐츠
    content_texts: List[str] = field(default_factory=list)  # 텍스트 블록 목록
    subsections: List['RestructuredSection'] = field(default_factory=list)


@dataclass
class RestructuredDocument:
    """재구성된 문서"""
    title: str
    sections: List[RestructuredSection]
    document_type: str
    main_topic: str


def restructure_document(
    parsed: ParsedDocument,
    plan: RestructurePlan,
    refine: bool = False,
) -> RestructuredDocument:
    """
    파싱된 문서를 재구성 계획에 따라 재구성합니다.

    Args:
        parsed: Docling으로 파싱된 문서 객체
        plan: 재구성 계획
        refine: True이면 LLM으로 내용을 다듬기

    Returns:
        RestructuredDocument: 재구성된 문서
    """
    print("[재구성] 문서 재구성 시작...")

    original_sections = parsed.sections
    used_sections: Set[int] = set()

    restructured_sections = _build_sections(
        plan.toc, original_sections, refine, used_sections
    )

    # 어디에도 배치되지 않은 원본 섹션 확인
    total_sections = len(original_sections)
    unused = [i for i in range(total_sections) if i not in used_sections]
    if unused:
        unused_texts = []
        unused_md_parts = []
        for idx in unused:
            section = original_sections[idx]
            if section.content_md.strip():
                unused_texts.extend(section.content_texts)
                unused_md_parts.append(section.content_md)

        if unused_texts:
            print(f"  [경고] {len(unused)}개 원본 섹션이 목차에 매핑되지 않아 '기타' 섹션에 추가")
            restructured_sections.append(RestructuredSection(
                level=1,
                title='기타',
                content_md='\n\n'.join(unused_md_parts),
                content_texts=unused_texts,
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
    original_sections: List[Section],
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
        own_indices = [
            idx for idx in item.source_sections
            if idx not in sub_source_indices
        ]
        content_md, content_texts = _collect_content(
            own_indices, original_sections, used_sections
        )

        # 빈 섹션에 description 삽입
        if not content_md.strip() and (item.description or item.source_sections):
            fallback_text = item.description if item.description else f"({item.title} 섹션)"
            content_md = fallback_text
            content_texts = [fallback_text]

        # 진단 로그
        print(f"  [매핑] '{item.title}' ← 인덱스 {own_indices} → {len(content_texts)}개 텍스트 블록"
              + (f" (하위: {len(item.subsections)}개)" if item.subsections else ""))

        # 내용 다듬기 (선택)
        if refine and content_md.strip():
            content_md, content_texts = _refine_content(item.title, content_md)

        # 하위 섹션 재귀 처리
        subsections = _build_sections(item.subsections, original_sections, refine, used_sections)

        sections.append(RestructuredSection(
            level=item.level,
            title=item.title,
            content_md=content_md,
            content_texts=content_texts,
            subsections=subsections,
        ))

    return sections


def _collect_all_source_indices(toc_items: List[TocItem], indices: set):
    """TOC 항목 트리에서 모든 source_sections 인덱스를 수집"""
    for item in toc_items:
        indices.update(item.source_sections)
        _collect_all_source_indices(item.subsections, indices)


def _collect_content(
    source_indices: List[int],
    original_sections: List[Section],
    used_sections: Set[int],
) -> tuple:
    """원본 섹션 인덱스에서 콘텐츠를 수집 (중복 방지)"""
    md_parts = []
    text_parts = []

    for idx in source_indices:
        if idx in used_sections:
            continue
        if 0 <= idx < len(original_sections):
            section = original_sections[idx]
            if section.content_md.strip():
                md_parts.append(section.content_md)
                text_parts.extend(section.content_texts)
            used_sections.add(idx)

    return '\n\n'.join(md_parts), text_parts


def _refine_content(title: str, content_md: str) -> tuple:
    """LLM으로 섹션 내용을 다듬기"""
    if len(content_md) < 50:
        return content_md, [content_md]

    try:
        client = get_ollama_client()
        prompt = f"""아래는 "{title}" 섹션의 내용입니다.
내용의 의미를 보존하면서 더 자연스럽고 읽기 쉽게 다듬어주세요.
원본 정보를 누락하거나 새로운 정보를 추가하지 마세요.
마크다운 형식을 유지하세요.
다듬어진 텍스트만 출력하세요.

[원본]
{content_md}"""

        res = client.chat(
            model=Config.MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options=get_llm_options(Config.TEMPERATURE_REFINE),
        )
        refined = res['message']['content'].strip()
        refined_texts = [p.strip() for p in refined.split('\n\n') if p.strip()]
        return refined, refined_texts

    except Exception as e:
        print(f"  [다듬기] 실패 (원본 유지): {e}")
        return content_md, [content_md]


def _count_sections(sections: List[RestructuredSection]) -> int:
    """총 섹션 수 카운트 (재귀)"""
    count = len(sections)
    for section in sections:
        count += _count_sections(section.subsections)
    return count
