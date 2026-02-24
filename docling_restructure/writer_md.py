"""
.md 출력 생성 모듈
재구성된 문서를 마크다운 파일로 출력합니다.
Docling의 마크다운 콘텐츠를 직접 활용합니다.
"""

import os
from typing import List

from .restructurer import RestructuredDocument, RestructuredSection


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
            if not section.content_md.strip() and not section.subsections:
                continue
            prefix = '  ' * indent
            anchor = _title_to_anchor(section.title)
            lines.append(f"{prefix}- [{section.title}](#{anchor})")
            _add_toc_entries(section.subsections, indent + 1)

    _add_toc_entries(sections)
    return lines


def _title_to_anchor(title: str) -> str:
    """마크다운 앵커 생성 (GitHub 스타일)"""
    anchor = title.lower().strip()
    cleaned = ''
    for ch in anchor:
        if ch.isalnum() or ch in (' ', '-', '_') or '\uac00' <= ch <= '\ud7a3':
            cleaned += ch
    return cleaned.replace(' ', '-')


def _write_section_md(section: RestructuredSection) -> List[str]:
    """섹션을 마크다운으로 변환"""
    if not section.content_md.strip() and not section.subsections:
        return []

    lines = []

    # 섹션 제목
    heading_prefix = '#' * min(section.level + 1, 6)
    lines.append(f"{heading_prefix} {section.title}")
    lines.append("")

    # 섹션 콘텐츠 (Docling의 마크다운을 직접 사용)
    if section.content_md.strip():
        lines.append(section.content_md)
        lines.append("")

    # 하위 섹션
    for subsection in section.subsections:
        sub_lines = _write_section_md(subsection)
        lines.extend(sub_lines)

    return lines
