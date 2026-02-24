"""
Docling 기반 문서 파싱 모듈
DOCX, PDF 등 다양한 형식의 문서를 Docling으로 파싱하여
구조화된 섹션 정보를 추출합니다.

비정형 문서(헤딩 없이 텍스트만 있는 문서)도 단락 기반으로 섹션을 분리합니다.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument, DocItemLabel


@dataclass
class Section:
    """파싱된 섹션"""
    index: int
    title: str
    level: int
    content_md: str         # 마크다운 형태의 내용 (LLM 분석 및 MD 출력용)
    content_texts: List[str] = field(default_factory=list)  # 텍스트 블록 목록
    table_data: List[dict] = field(default_factory=list)     # 테이블 데이터
    has_tables: bool = False
    has_lists: bool = False
    char_count: int = 0


@dataclass
class ParsedDocument:
    """파싱된 문서 전체"""
    markdown_text: str       # 전체 마크다운 (LLM 입력용)
    sections: List[Section]
    title: str
    source_format: str       # 'docx', 'pdf' 등
    is_structured: bool      # 헤딩이 있는 구조화된 문서인지


# 섹션으로 분리할 때 사용하는 최소 단락 수 (비정형 문서)
_MIN_PARAGRAPHS_PER_SECTION = 3
# 비정형 문서에서 섹션 분리 시 최대 글자 수
_MAX_CHARS_PER_SECTION = 2000


def parse_document(file_path: str) -> ParsedDocument:
    """
    Docling으로 문서를 파싱하고 구조화된 표현을 반환합니다.

    Args:
        file_path: 문서 파일 경로 (.docx, .pdf 등)

    Returns:
        ParsedDocument: 파싱된 문서 객체
    """
    ext = os.path.splitext(file_path)[1].lower()
    print(f"[파싱] Docling으로 {ext} 파일 파싱 중: {file_path}")

    converter = DocumentConverter()
    result = converter.convert(file_path)
    doc: DoclingDocument = result.document

    # 마크다운 내보내기 (LLM 분석용)
    markdown_text = doc.export_to_markdown()
    print(f"  → 마크다운 변환 완료: {len(markdown_text)}자")

    # 구조 분석: 헤딩이 있는지 확인
    headings = _find_headings(doc)
    is_structured = len(headings) > 0

    if is_structured:
        print(f"  → 구조화된 문서: {len(headings)}개 헤딩 감지")
        sections = _extract_sections_structured(doc, headings)
    else:
        print("  → 비정형 문서: 헤딩 없음, 단락 기반 섹션 분리")
        sections = _extract_sections_unstructured(doc)

    # 문서 제목 추출
    title = _detect_title(doc, sections)

    print(f"  → {len(sections)}개 섹션 추출 완료")

    return ParsedDocument(
        markdown_text=markdown_text,
        sections=sections,
        title=title,
        source_format=ext.lstrip('.'),
        is_structured=is_structured,
    )


def _find_headings(doc: DoclingDocument) -> list:
    """DoclingDocument에서 모든 헤딩 아이템을 찾습니다."""
    headings = []
    for item, _level in doc.iterate_items():
        if hasattr(item, 'label'):
            label = item.label
            if label in (
                DocItemLabel.SECTION_HEADER,
                DocItemLabel.TITLE,
                DocItemLabel.PAGE_HEADER,
            ):
                text = item.text if hasattr(item, 'text') else str(item)
                text = text.strip()
                if text:
                    # 헤딩 레벨 결정
                    level = _determine_heading_level(item, label)
                    headings.append({
                        'text': text,
                        'level': level,
                        'item': item,
                    })
    return headings


def _determine_heading_level(item, label) -> int:
    """Docling 아이템에서 헤딩 레벨을 결정합니다."""
    if label == DocItemLabel.TITLE:
        return 0  # 문서 제목

    # Docling의 level 속성이 있으면 사용
    if hasattr(item, 'level') and item.level is not None:
        return max(1, item.level)

    # 기본값: 섹션 헤더는 level 1
    return 1


def _extract_sections_structured(doc: DoclingDocument, headings: list) -> List[Section]:
    """구조화된 문서에서 헤딩 기반으로 섹션을 추출합니다."""
    # 마크다운을 줄 단위로 분석하여 섹션 분리
    md_text = doc.export_to_markdown()
    lines = md_text.split('\n')

    sections = []
    current_title = '서두'
    current_level = 0
    current_lines = []
    current_texts = []
    current_tables = []
    current_has_tables = False
    current_has_lists = False

    in_table = False

    for line in lines:
        # 마크다운 헤딩 감지
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            # 이전 섹션 저장
            if current_lines or current_title != '서두':
                content_md = '\n'.join(current_lines).strip()
                sections.append(Section(
                    index=len(sections),
                    title=current_title,
                    level=current_level,
                    content_md=content_md,
                    content_texts=current_texts[:],
                    table_data=current_tables[:],
                    has_tables=current_has_tables,
                    has_lists=current_has_lists,
                    char_count=len(content_md),
                ))

            # 새 섹션 시작
            level = len(heading_match.group(1))
            current_title = heading_match.group(2).strip()
            current_level = level
            current_lines = []
            current_texts = []
            current_tables = []
            current_has_tables = False
            current_has_lists = False
            continue

        # 내용 수집
        current_lines.append(line)

        stripped = line.strip()
        if stripped:
            # 테이블 감지
            if stripped.startswith('|') and '|' in stripped[1:]:
                current_has_tables = True
                in_table = True
            elif in_table:
                in_table = False

            # 리스트 감지
            if re.match(r'^[-*+]\s', stripped) or re.match(r'^\d+[.)]\s', stripped):
                current_has_lists = True

            # 텍스트 블록 수집 (테이블 구분선 제외)
            if not re.match(r'^[|\-\s:]+$', stripped):
                current_texts.append(stripped)

    # 마지막 섹션 저장
    content_md = '\n'.join(current_lines).strip()
    if content_md or current_title != '서두':
        sections.append(Section(
            index=len(sections),
            title=current_title,
            level=current_level,
            content_md=content_md,
            content_texts=current_texts,
            table_data=current_tables,
            has_tables=current_has_tables,
            has_lists=current_has_lists,
            char_count=len(content_md),
        ))

    # 서두 섹션이 비어있으면 제거
    if sections and sections[0].title == '서두' and not sections[0].content_md.strip():
        sections.pop(0)
        for i, s in enumerate(sections):
            s.index = i

    return sections


def _extract_sections_unstructured(doc: DoclingDocument) -> List[Section]:
    """
    비정형 문서(헤딩 없음)에서 단락 기반으로 섹션을 분리합니다.

    긴 텍스트를 의미 단위로 분리하여 LLM이 재구조화할 수 있도록 합니다.
    """
    md_text = doc.export_to_markdown()

    # 빈 줄로 단락 분리
    paragraphs = []
    current = []
    for line in md_text.split('\n'):
        if line.strip():
            current.append(line)
        elif current:
            paragraphs.append('\n'.join(current))
            current = []
    if current:
        paragraphs.append('\n'.join(current))

    if not paragraphs:
        return [Section(
            index=0,
            title='본문',
            level=1,
            content_md=md_text.strip(),
            content_texts=[md_text.strip()],
            char_count=len(md_text.strip()),
        )]

    # 단락들을 적절한 크기의 섹션으로 그룹핑
    sections = []
    current_paras = []
    current_chars = 0

    for para in paragraphs:
        current_paras.append(para)
        current_chars += len(para)

        # 섹션 분리 조건: 최소 단락 수 이상 + 최대 글자 수 초과
        if (len(current_paras) >= _MIN_PARAGRAPHS_PER_SECTION
                and current_chars >= _MAX_CHARS_PER_SECTION):
            content_md = '\n\n'.join(current_paras)
            has_tables = any('|' in p and p.strip().startswith('|') for p in current_paras)
            has_lists = any(
                re.match(r'^[-*+]\s', p.strip()) or re.match(r'^\d+[.)]\s', p.strip())
                for p in current_paras
            )

            # 첫 줄을 제목으로 사용 (최대 50자)
            first_line = current_paras[0].split('\n')[0].strip()
            title = first_line[:50] + ('...' if len(first_line) > 50 else '')

            sections.append(Section(
                index=len(sections),
                title=f"단락 {len(sections)+1}: {title}",
                level=1,
                content_md=content_md,
                content_texts=[p for p in current_paras],
                has_tables=has_tables,
                has_lists=has_lists,
                char_count=current_chars,
            ))
            current_paras = []
            current_chars = 0

    # 남은 단락 처리
    if current_paras:
        content_md = '\n\n'.join(current_paras)
        has_tables = any('|' in p and p.strip().startswith('|') for p in current_paras)
        has_lists = any(
            re.match(r'^[-*+]\s', p.strip()) or re.match(r'^\d+[.)]\s', p.strip())
            for p in current_paras
        )
        first_line = current_paras[0].split('\n')[0].strip()
        title = first_line[:50] + ('...' if len(first_line) > 50 else '')

        sections.append(Section(
            index=len(sections),
            title=f"단락 {len(sections)+1}: {title}",
            level=1,
            content_md=content_md,
            content_texts=[p for p in current_paras],
            has_tables=has_tables,
            has_lists=has_lists,
            char_count=current_chars,
        ))

    return sections


def _detect_title(doc: DoclingDocument, sections: List[Section]) -> str:
    """문서 제목을 감지합니다."""
    # Docling이 감지한 TITLE 라벨 사용
    for item, _level in doc.iterate_items():
        if hasattr(item, 'label') and item.label == DocItemLabel.TITLE:
            text = item.text if hasattr(item, 'text') else str(item)
            if text.strip():
                return text.strip()

    # 마크다운에서 첫 번째 H1 헤딩 사용
    md = doc.export_to_markdown()
    match = re.search(r'^#\s+(.+)$', md, re.MULTILINE)
    if match:
        return match.group(1).strip()

    # 첫 번째 섹션 제목 사용
    if sections:
        title = sections[0].title
        if title not in ('서두', '본문') and not title.startswith('단락 '):
            return title

    return '제목 없음'


def sections_to_text_summary(sections: List[Section]) -> str:
    """섹션 리스트를 LLM 분석용 텍스트 요약으로 변환"""
    lines = []
    for s in sections:
        indent = '  ' * max(0, s.level - 1)
        meta_parts = [f"{s.char_count}자"]
        if s.has_tables:
            meta_parts.append("테이블 포함")
        if s.has_lists:
            meta_parts.append("리스트 포함")
        meta = ', '.join(meta_parts)

        # 내용 미리보기 (최대 200자)
        preview = s.content_md[:200].replace('\n', ' ').strip()
        if len(s.content_md) > 200:
            preview += '...'

        lines.append(f"[{s.index}] {indent}{s.title} ({meta}, 미리보기: {preview})")
    return "\n".join(lines)
