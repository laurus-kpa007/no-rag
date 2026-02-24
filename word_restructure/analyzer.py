"""
LLM 기반 문서 분석 및 재구성 계획 생성 모듈
Ollama gemma3:27b를 사용하여 문서를 분석하고 최적의 구조를 제안합니다.
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from .config import Config, get_ollama_client, get_llm_options
from .extractor import DocumentElement, ExtractedDocument, elements_to_sections, sections_to_text_summary


@dataclass
class TocItem:
    """목차 항목"""
    level: int
    title: str
    source_sections: List[int] = field(default_factory=list)  # 매핑할 원본 섹션 인덱스
    subsections: List['TocItem'] = field(default_factory=list)
    description: str = ''  # LLM이 제안한 섹션 설명


@dataclass
class DocumentAnalysis:
    """문서 분석 결과"""
    document_type: str
    main_topic: str
    key_themes: List[str]
    current_structure_assessment: str
    content_sections: List[dict]


@dataclass
class RestructurePlan:
    """문서 재구성 계획"""
    title: str
    toc: List[TocItem]
    analysis: DocumentAnalysis


def _call_llm(prompt: str, temperature: float = None) -> str:
    """LLM 호출 헬퍼"""
    client = get_ollama_client()
    res = client.chat(
        model=Config.MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        options=get_llm_options(temperature),
    )
    return res['message']['content'].strip()


def _call_llm_json(prompt: str, temperature: float = None) -> dict:
    """LLM 호출 후 JSON 파싱"""
    client = get_ollama_client()
    res = client.chat(
        model=Config.MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        options=get_llm_options(temperature),
        format='json',
    )
    content = res['message']['content'].strip()
    return json.loads(content)


def _truncate_text(text: str, max_chars: int = None) -> str:
    """텍스트를 최대 길이로 자르기"""
    max_chars = max_chars or Config.MAX_CONTENT_CHARS
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... (전체 {len(text)}자 중 {max_chars}자까지 표시)"


def analyze_document(extracted: ExtractedDocument) -> DocumentAnalysis:
    """
    Pass 1: 문서의 현재 구조를 분석합니다.

    Args:
        extracted: 추출된 문서 객체

    Returns:
        DocumentAnalysis: 문서 분석 결과
    """
    print("[분석] Pass 1: 문서 구조 분석 중...")

    doc_text = _truncate_text(extracted.raw_text)

    prompt = f"""당신은 전문 문서 분석가입니다. 아래 문서의 내용과 구조를 분석하세요.

[문서 내용]
{doc_text}

다음 정보를 JSON 형식으로 출력하세요. 반드시 유효한 JSON만 출력하세요:
{{
  "document_type": "문서 유형 (보고서/매뉴얼/제안서/회의록/기술문서/논문/기타 중 하나)",
  "main_topic": "문서의 핵심 주제 (한 문장)",
  "key_themes": ["주요 테마1", "주요 테마2", "주요 테마3"],
  "current_structure_assessment": "현재 문서 구조에 대한 평가와 개선 필요 사항",
  "content_sections": [
    {{"index": 0, "topic": "주제", "summary": "핵심 내용 요약 (2-3문장)", "importance": "high/medium/low"}}
  ]
}}

규칙:
- content_sections에는 문서의 논리적 내용 블록을 모두 포함하세요
- 각 블록의 index는 0부터 순차적으로
- summary는 해당 블록의 핵심 내용을 간결하게 요약
- 문서의 언어에 맞춰 응답하세요 (한국어 문서면 한국어로)"""

    try:
        result = _call_llm_json(prompt)
        return DocumentAnalysis(
            document_type=result.get('document_type', '기타'),
            main_topic=result.get('main_topic', extracted.title),
            key_themes=result.get('key_themes', []),
            current_structure_assessment=result.get('current_structure_assessment', ''),
            content_sections=result.get('content_sections', []),
        )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[분석] Pass 1 JSON 파싱 실패, 재시도: {e}")
        # 폴백: 간단한 분석
        return DocumentAnalysis(
            document_type='기타',
            main_topic=extracted.title,
            key_themes=[],
            current_structure_assessment='자동 분석 실패 - 기본 구조로 진행',
            content_sections=[],
        )


def generate_restructure_plan(
    extracted: ExtractedDocument,
    analysis: DocumentAnalysis,
) -> RestructurePlan:
    """
    Pass 2: 문서에 최적화된 재구성 계획을 생성합니다.

    Args:
        extracted: 추출된 문서 객체
        analysis: Pass 1의 분석 결과

    Returns:
        RestructurePlan: 재구성 계획
    """
    print("[분석] Pass 2: 재구성 계획 생성 중...")

    # 원본 섹션 정보 구성
    sections = elements_to_sections(extracted.elements)
    sections_text = sections_to_text_summary(sections)

    # 분석 결과 요약
    analysis_summary = json.dumps({
        'document_type': analysis.document_type,
        'main_topic': analysis.main_topic,
        'key_themes': analysis.key_themes,
        'assessment': analysis.current_structure_assessment,
    }, ensure_ascii=False, indent=2)

    # 문서 규모 기반 동적 가이드 계산
    n = len(sections)
    if n <= 5:
        min_major = 2
        depth_guide = "1~2 레벨 구조로 구성하세요."
        detail_guide = ""
    elif n <= 15:
        min_major = max(3, n // 4)
        depth_guide = "반드시 2레벨(대분류/중분류) 이상으로 구성하세요."
        detail_guide = (
            f"- 대분류(level 1)는 최소 {min_major}개 이상 만드세요\n"
            f"- 원본 섹션이 3개 이상 배정된 대분류는 반드시 중분류(subsections)로 세분화하세요"
        )
    else:
        min_major = max(5, n // 5)
        depth_guide = "반드시 3레벨(대분류/중분류/소분류) 구조로 상세하게 구성하세요."
        detail_guide = (
            f"- 대분류(level 1)는 최소 {min_major}개 이상 만드세요\n"
            f"- 모든 대분류는 반드시 2개 이상의 중분류(level 2 subsections)를 포함하세요\n"
            f"- 원본 섹션이 3개 이상 배정된 중분류는 소분류(level 3)로 추가 세분화하세요"
        )

    print(f"  [가이드] 원본 {n}개 섹션 → 최소 대분류 {min_major}개, {depth_guide}")

    prompt = f"""당신은 전문 문서 구조화 전문가입니다. 문서를 체계적이고 상세한 트리 구조 목차로 재구성합니다.
이 목차는 RAG(Retrieval-Augmented Generation) 시스템의 PageIndex 전처리에 사용됩니다.
따라서 **정확하고 균형 잡힌 트리 구조**가 매우 중요합니다.

[문서 분석 결과]
{analysis_summary}

[원본 섹션 목록] (인덱스: 0 ~ {n-1}, 총 {n}개)
{sections_text}

위 문서를 가장 적합한 구조로 재구성하는 **상세한 목차**를 설계하세요.
{depth_guide}

JSON 형식으로 출력하세요. 반드시 유효한 JSON만 출력하세요:
{{
  "title": "재구성된 문서 제목",
  "toc": [
    {{
      "level": 1,
      "title": "1. 개요",
      "description": "문서의 배경과 목적을 소개",
      "source_sections": [0, 1],
      "subsections": [
        {{
          "level": 2,
          "title": "1.1 배경",
          "description": "프로젝트 배경 설명",
          "source_sections": [0]
        }},
        {{
          "level": 2,
          "title": "1.2 목적 및 범위",
          "description": "문서의 목적과 범위 정의",
          "source_sections": [1]
        }}
      ]
    }},
    {{
      "level": 1,
      "title": "2. 현황 분석",
      "description": "현재 상황에 대한 분석 결과",
      "source_sections": [2, 3, 4, 5],
      "subsections": [
        {{
          "level": 2,
          "title": "2.1 데이터 분석",
          "description": "수집된 데이터의 분석 결과",
          "source_sections": [2, 3]
        }},
        {{
          "level": 2,
          "title": "2.2 문제점 도출",
          "description": "분석을 통해 도출된 주요 문제점",
          "source_sections": [4]
        }},
        {{
          "level": 2,
          "title": "2.3 시사점",
          "description": "분석 결과의 시사점 정리",
          "source_sections": [5]
        }}
      ]
    }},
    {{
      "level": 1,
      "title": "3. 세부 내용",
      "description": "핵심 내용 상세 기술",
      "source_sections": [6, 7, 8],
      "subsections": [
        {{
          "level": 2,
          "title": "3.1 주요 항목",
          "description": "핵심 항목에 대한 상세 설명",
          "source_sections": [6, 7]
        }},
        {{
          "level": 2,
          "title": "3.2 부가 항목",
          "description": "보조적인 항목 설명",
          "source_sections": [8]
        }}
      ]
    }},
    {{
      "level": 1,
      "title": "4. 결론 및 제언",
      "description": "결론 요약과 향후 제언",
      "source_sections": [9, 10],
      "subsections": [
        {{
          "level": 2,
          "title": "4.1 결론",
          "description": "핵심 결론 요약",
          "source_sections": [9]
        }},
        {{
          "level": 2,
          "title": "4.2 향후 제언",
          "description": "향후 방향에 대한 제언",
          "source_sections": [10]
        }}
      ]
    }}
  ]
}}

중요 규칙:
1. source_sections는 반드시 0 ~ {n-1} 범위의 정수만 사용하세요 (총 {n}개 섹션)
2. 모든 원본 섹션(0~{n-1})이 반드시 하나 이상의 source_sections에 포함되어야 합니다 (내용 누락 방지)
3. 하위 섹션(subsections)의 source_sections는 상위 섹션의 source_sections 부분집합이어야 합니다
   - 예: 상위 source_sections=[0,1,2] → 하위1=[0], 하위2=[1,2] (상위 본문에는 나머지만 배치됨)
4. source_sections가 비어있으면([]) 해당 섹션에 원본 내용이 배치되지 않으므로, 반드시 하나 이상의 인덱스를 넣으세요
5. description은 반드시 작성하세요 (빈 문자열 금지)

목차 세분화 규칙 (매우 중요!):
{detail_guide if detail_guide else "- 문서 내용에 맞게 적절히 세분화하세요"}
- 대분류를 2~3개만 만들고 끝내지 마세요. 내용이 많은 문서는 충분히 세분화해야 합니다
- 하나의 대분류에 원본 섹션을 과도하게 몰아넣지 마세요. 논리적으로 분산시키세요
- 내용이 많은 대분류(source_sections 3개 이상)는 반드시 중분류(subsections)로 나누세요
- level 1(대분류)은 최소 {min_major}개 이상 생성하세요

RAG 트리 품질 규칙 (이 목차는 검색 인덱스로 사용됩니다):
- 각 리프 노드(최하위 섹션)는 명확하고 구체적인 주제를 가져야 합니다
- 트리가 한쪽으로 치우치지 않게 균형 잡힌 구조를 만드세요 (한 대분류에 8개 중분류, 다른 대분류에 1개 중분류 등은 지양)
- 각 섹션의 title은 해당 내용을 검색할 때 키워드로 활용되므로, 구체적이고 명확하게 작성하세요
- 너무 포괄적인 제목(예: "기타", "추가 내용") 대신 내용을 반영하는 제목을 사용하세요
- description은 해당 섹션의 핵심 내용을 요약하는 1-2문장으로 작성하세요 (검색 컨텍스트로 활용됨)

구조 규칙:
- 문서 유형({analysis.document_type})에 적합한 표준 구조를 따르세요
- level은 1부터 시작 (1=대분류, 2=중분류, 3=소분류), 최대 3레벨
- 문서 유형별 표준 구조 (중분류까지 예시):
  - 보고서: 1.요약(1.1 배경, 1.2 핵심결과) → 2.서론(2.1 목적, 2.2 범위) → 3.본론(3.1 분석, 3.2 결과, 3.3 논의) → 4.결론(4.1 요약, 4.2 제언) → 5.부록
  - 매뉴얼: 1.개요(1.1 소개, 1.2 구성) → 2.설치(2.1 요구사항, 2.2 절차) → 3.사용법(3.1 기본, 3.2 고급) → 4.FAQ → 5.부록
  - 제안서: 1.요약 → 2.배경(2.1 현황, 2.2 문제점) → 3.제안(3.1 개요, 3.2 세부방안) → 4.기대효과(4.1 정량, 4.2 정성) → 5.일정/예산
  - 기술문서: 1.개요(1.1 목적, 1.2 범위) → 2.아키텍처(2.1 구조, 2.2 흐름) → 3.상세설명(3.1~3.N 각 모듈) → 4.API/참조 → 5.부록
  - 기타: 1.서론(1.1 배경, 1.2 목적) → 2~N.본론(각각 중분류 포함) → N+1.결론
- 문서의 언어에 맞춰 응답하세요"""

    try:
        result = _call_llm_json(prompt, temperature=Config.TEMPERATURE_STRUCTURE)
        toc_items = _parse_toc_items(result.get('toc', []))

        total_sections = len(sections)

        # 인덱스 유효성 검증 (범위 초과 제거, 경고 출력)
        _validate_source_sections(toc_items, total_sections)

        # 누락된 원본 섹션 확인 및 보정
        all_mapped = set()
        _collect_mapped_sections(toc_items, all_mapped)
        unmapped = [i for i in range(total_sections) if i not in all_mapped]

        if unmapped:
            print(f"  [보정] 매핑되지 않은 원본 섹션 {unmapped}을 '기타'에 추가")
            toc_items.append(TocItem(
                level=1,
                title='기타',
                source_sections=unmapped,
                description='분류되지 않은 원본 내용',
            ))

        return RestructurePlan(
            title=result.get('title', extracted.title),
            toc=toc_items,
            analysis=analysis,
        )
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[분석] Pass 2 JSON 파싱 실패, 기본 구조 사용: {e}")
        return _create_fallback_plan(extracted, analysis, sections)


def _parse_toc_items(toc_data: list) -> List[TocItem]:
    """JSON 목차 데이터를 TocItem 리스트로 변환"""
    items = []
    for item_data in toc_data:
        subsections = _parse_toc_items(item_data.get('subsections', []))
        items.append(TocItem(
            level=item_data.get('level', 1),
            title=item_data.get('title', ''),
            source_sections=item_data.get('source_sections', []),
            subsections=subsections,
            description=item_data.get('description', ''),
        ))
    return items


def _validate_source_sections(toc_items: List[TocItem], total_sections: int):
    """source_sections의 유효성 검증 및 보정"""
    for item in toc_items:
        # 범위 초과 인덱스 제거
        valid = [i for i in item.source_sections if isinstance(i, int) and 0 <= i < total_sections]
        invalid = [i for i in item.source_sections if i not in valid]
        if invalid:
            print(f"  [경고] '{item.title}' 범위 초과 인덱스 제거: {invalid} (유효: 0~{total_sections-1})")
        item.source_sections = valid

        # 하위 섹션도 재귀 검증
        _validate_source_sections(item.subsections, total_sections)


def _collect_mapped_sections(toc_items: List[TocItem], mapped: set):
    """목차에 매핑된 모든 원본 섹션 인덱스를 수집"""
    for item in toc_items:
        mapped.update(item.source_sections)
        _collect_mapped_sections(item.subsections, mapped)


def _create_fallback_plan(
    extracted: ExtractedDocument,
    analysis: DocumentAnalysis,
    sections: list,
) -> RestructurePlan:
    """LLM 응답 실패 시 기본 재구성 계획 생성"""
    toc_items = []
    for i, section in enumerate(sections):
        toc_items.append(TocItem(
            level=section['level'] if section['level'] > 0 else 1,
            title=section['title'],
            source_sections=[i],
        ))

    if not toc_items:
        toc_items.append(TocItem(
            level=1,
            title='본문',
            source_sections=list(range(len(sections))),
        ))

    return RestructurePlan(
        title=extracted.title,
        toc=toc_items,
        analysis=analysis,
    )


def plan_to_text(plan: RestructurePlan) -> str:
    """재구성 계획을 사람이 읽기 쉬운 텍스트로 변환 (진행 상황 출력용)"""
    lines = [f"문서 제목: {plan.title}"]
    lines.append(f"문서 유형: {plan.analysis.document_type}")
    lines.append(f"주요 주제: {plan.analysis.main_topic}")
    lines.append(f"\n목차:")

    def _format_toc(items: List[TocItem], indent: int = 0):
        for item in items:
            prefix = '  ' * indent
            src = f" (원본 섹션: {item.source_sections})" if item.source_sections else ""
            lines.append(f"{prefix}{'#' * item.level} {item.title}{src}")
            if item.description:
                lines.append(f"{prefix}  └ {item.description}")
            _format_toc(item.subsections, indent + 1)

    _format_toc(plan.toc)
    return "\n".join(lines)
