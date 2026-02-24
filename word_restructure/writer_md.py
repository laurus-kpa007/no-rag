"""
.md 출력 생성 모듈
재구성된 문서를 마크다운 파일로 출력합니다.
"""

import os
from typing import List

from .restructurer import RestructuredDocument, RestructuredSection
from .extractor import DocumentElement


def write_md(restructured: RestructuredDocument, output_path: str):
    """
    재구성된 문서를 .md 파일로 출력합니다.

    Args:
        restructured: 재구성된 문서 객체
        output_path: 출력 파일 경로
    """
    print(f"[출력] .md 파일 생성 중: {output_path}")

    lines = []

    # 문서 제목
    lines.append(f"# {restructured.title}")
    lines.append("")
    lines.append(f"> 문서 유형: {restructured.document_type} | 주제: {restructured.main_topic}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 자동 목차
    lines.append("## 목차")
    lines.append("")
    toc_lines = _generate_toc(restructured.sections)
    lines.extend(toc_lines)
    lines.append("")
    lines.append("---")
    lines.append("")

    # 본문
    for section in restructured.sections:
        section_lines = _write_section_md(section)
        lines.extend(section_lines)

    content = "\n".join(lines)

    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"[출력] .md 파일 저장 완료: {output_path}")


def _generate_toc(sections: List[RestructuredSection]) -> List[str]:
    """목차 생성"""
    lines = []

    def _add_toc_entries(sects: List[RestructuredSection], indent: int = 0):
        for section in sects:
            prefix = '  ' * indent
            anchor = _title_to_anchor(section.title)
            lines.append(f"{prefix}- [{section.title}](#{anchor})")
            _add_toc_entries(section.subsections, indent + 1)

    _add_toc_entries(sections)
    return lines


def _title_to_anchor(title: str) -> str:
    """마크다운 앵커 생성 (GitHub 스타일)"""
    anchor = title.lower().strip()
    # 특수문자 제거 (한글, 영문, 숫자, 공백, 하이픈만 유지)
    cleaned = ''
    for ch in anchor:
        if ch.isalnum() or ch in (' ', '-', '_') or '\uac00' <= ch <= '\ud7a3':
            cleaned += ch
    return cleaned.replace(' ', '-')


def _write_section_md(section: RestructuredSection) -> List[str]:
    """섹션을 마크다운으로 변환"""
    # 내용도 하위 섹션도 없는 빈 섹션은 건너뜀
    if not section.content_elements and not section.subsections:
        return []

    lines = []

    # 섹션 제목
    heading_prefix = '#' * min(section.level + 1, 6)  # 문서 제목이 #이므로 +1
    lines.append(f"{heading_prefix} {section.title}")
    lines.append("")

    # 섹션 내용
    prev_type = None
    for elem in section.content_elements:
        # 리스트 → 비리스트 전환 시 빈 줄 추가
        if prev_type == 'list_item' and elem.type != 'list_item':
            lines.append("")
        elem_lines = _write_element_md(elem)
        lines.extend(elem_lines)
        prev_type = elem.type

    # 마지막이 리스트였으면 빈 줄 추가
    if prev_type == 'list_item':
        lines.append("")

    # 하위 섹션
    for subsection in section.subsections:
        sub_lines = _write_section_md(subsection)
        lines.extend(sub_lines)

    return lines


def _write_element_md(elem: DocumentElement) -> List[str]:
    """개별 요소를 마크다운으로 변환"""
    lines = []

    if elem.type == 'paragraph':
        lines.append(elem.content)
        lines.append("")

    elif elem.type == 'list_item':
        indent = '  ' * elem.level
        lines.append(f"{indent}- {elem.content}")
        # 리스트 아이템은 연속으로 올 수 있으므로 빈 줄은 마지막에만

    elif elem.type == 'table':
        table_lines = _write_table_md(elem)
        lines.extend(table_lines)

    elif elem.type == 'heading':
        # 남은 heading은 bold 텍스트로
        lines.append(f"**{elem.content}**")
        lines.append("")

    return lines


def _write_table_md(elem: DocumentElement) -> List[str]:
    """표를 마크다운 테이블로 변환"""
    lines = []
    table_data = elem.metadata

    if not table_data or 'rows' not in table_data:
        # 메타데이터가 없으면 원본 텍스트를 코드 블록으로
        lines.append("```")
        lines.append(elem.content)
        lines.append("```")
        lines.append("")
        return lines

    rows = table_data['rows']
    if not rows:
        return lines

    num_cols = max(len(row) for row in rows)

    # 열 너비 계산 (최소 3자)
    col_widths = [3] * num_cols
    for row in rows:
        for j, cell in enumerate(row):
            if j < num_cols:
                col_widths[j] = max(col_widths[j], len(cell))

    # 헤더 행
    header = rows[0]
    header_cells = []
    for j in range(num_cols):
        cell = header[j] if j < len(header) else ''
        header_cells.append(f" {cell.ljust(col_widths[j])} ")
    lines.append("|" + "|".join(header_cells) + "|")

    # 구분선
    separator_cells = ["-" * (col_widths[j] + 2) for j in range(num_cols)]
    lines.append("|" + "|".join(separator_cells) + "|")

    # 데이터 행
    for row in rows[1:]:
        data_cells = []
        for j in range(num_cols):
            cell = row[j] if j < len(row) else ''
            data_cells.append(f" {cell.ljust(col_widths[j])} ")
        lines.append("|" + "|".join(data_cells) + "|")

    lines.append("")
    return lines
