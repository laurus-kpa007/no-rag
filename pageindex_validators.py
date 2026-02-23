"""
PageIndex Ultra-Precision Validation Module

이 모듈은 검색 결과의 품질을 검증하는 다양한 검증 함수를 제공합니다.

주요 기능:
1. 증거 검증 (Evidence Validation): 답변의 근거가 실제 문서에 존재하는지 확인
2. 모순 검사 (Contradiction Detection): 섹션 간 상충되는 정보 탐지
3. 완전성 검사 (Completeness Check): 질문에 충분한 답변이 가능한지 확인
4. 신뢰성 평가 (Credibility Assessment): 정보의 신뢰성 평가
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import re


@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    confidence: float  # 0-1
    issues: List[str]  # 발견된 문제점
    recommendations: List[str]  # 개선 제안


class EvidenceValidator:
    """증거 검증기"""

    @staticmethod
    def validate_claim_with_context(claim: str, context: str, threshold: float = 0.6) -> bool:
        """
        주장이 컨텍스트에서 지원되는지 검증

        Args:
            claim: 검증할 주장
            context: 참조 컨텍스트
            threshold: 최소 단어 중첩 비율 (0-1)

        Returns:
            bool: 검증 통과 여부
        """
        # 주장의 주요 키워드 추출 (3글자 이상)
        claim_words = set(word.lower() for word in re.findall(r'\w+', claim) if len(word) > 2)
        context_words = set(word.lower() for word in re.findall(r'\w+', context) if len(word) > 2)

        if not claim_words:
            return False

        # 중첩 비율 계산
        overlap = len(claim_words & context_words)
        ratio = overlap / len(claim_words)

        return ratio >= threshold

    @staticmethod
    def extract_evidence_sentences(text: str, section_delimiter: str = "[") -> List[Tuple[str, str]]:
        """
        텍스트에서 증거 문장과 출처 추출

        Args:
            text: 분석할 텍스트
            section_delimiter: 섹션 구분자

        Returns:
            List[(문장, 출처)]: 추출된 증거 문장과 출처 쌍
        """
        evidence_pairs = []
        current_section = "Unknown"

        for line in text.split('\n'):
            # 섹션 제목 감지
            if line.strip().startswith(section_delimiter):
                current_section = line.strip().strip('[]')
            # 증거 문장 감지 (-, •, 숫자. 등으로 시작)
            elif re.match(r'^[\s\-•\d.]+', line.strip()):
                sentence = line.strip().lstrip('-•').lstrip()
                if sentence:
                    evidence_pairs.append((sentence, current_section))

        return evidence_pairs

    @staticmethod
    def verify_evidence_batch(evidence_pairs: List[Tuple[str, str]],
                              sections: Dict[str, str],
                              threshold: float = 0.6) -> List[bool]:
        """
        증거 문장 배치 검증

        Args:
            evidence_pairs: [(문장, 섹션명), ...] 리스트
            sections: {섹션명: 내용} 딕셔너리
            threshold: 검증 임계값

        Returns:
            List[bool]: 각 증거의 검증 결과
        """
        results = []

        for sentence, section_name in evidence_pairs:
            # 정확한 섹션명 매칭 또는 부분 매칭
            matched_content = None
            for sec_name, content in sections.items():
                if section_name.lower() in sec_name.lower() or sec_name.lower() in section_name.lower():
                    matched_content = content
                    break

            if matched_content:
                is_valid = EvidenceValidator.validate_claim_with_context(
                    sentence, matched_content, threshold)
                results.append(is_valid)
            else:
                # 섹션을 찾지 못하면 실패
                results.append(False)

        return results


class ContradictionDetector:
    """모순 검사기"""

    @staticmethod
    def detect_numerical_contradictions(sections: List[Tuple[str, str]]) -> List[str]:
        """
        숫자 정보의 모순 검사

        예: 섹션 A에서 "가격: 10,000원", 섹션 B에서 "가격: 15,000원"

        Args:
            sections: [(제목, 내용), ...] 리스트

        Returns:
            List[str]: 발견된 모순 목록
        """
        contradictions = []

        # 숫자와 단위 추출 (간단한 패턴 매칭)
        number_pattern = re.compile(r'(\d+[,\d]*)\s*([원달러유로%개명시간일년월])')

        section_numbers = {}
        for title, content in sections:
            matches = number_pattern.findall(content)
            section_numbers[title] = matches

        # 동일한 단위의 숫자가 다르면 모순 가능성
        unit_values = {}  # {단위: [(섹션명, 값), ...]}
        for title, matches in section_numbers.items():
            for value, unit in matches:
                value_clean = value.replace(',', '')
                if unit not in unit_values:
                    unit_values[unit] = []
                unit_values[unit].append((title, int(value_clean)))

        # 같은 단위에서 값이 다르면 모순
        for unit, values in unit_values.items():
            if len(values) > 1:
                unique_values = set(v for _, v in values)
                if len(unique_values) > 1:
                    sections_str = ", ".join(f"{t}({v})" for t, v in values)
                    contradictions.append(f"숫자 모순 ({unit}): {sections_str}")

        return contradictions

    @staticmethod
    def detect_semantic_contradictions_with_llm(sections: List[Tuple[str, str]],
                                                 client,
                                                 max_pairs: int = 5) -> List[str]:
        """
        LLM을 사용한 의미적 모순 검사

        Args:
            sections: [(제목, 내용), ...] 리스트
            client: Ollama 클라이언트
            max_pairs: 검사할 최대 섹션 쌍 수

        Returns:
            List[str]: 발견된 모순 목록
        """
        from itertools import combinations

        contradictions = []

        # 모든 조합을 검사하는 것은 비효율적이므로 샘플링
        for (t1, c1), (t2, c2) in list(combinations(sections, 2))[:max_pairs]:
            prompt = f"""다음 두 섹션을 비교하여 **상충되거나 모순되는 정보**가 있는지 확인하세요:

섹션 1: {t1}
{c1[:600]}...

섹션 2: {t2}
{c2[:600]}...

모순이 있다면:
- "모순 발견: [구체적인 내용]"

모순이 없다면:
- "모순 없음"

한 줄로만 답하세요."""

            try:
                response = client.chat(model=client.model, messages=[
                    {'role': 'user', 'content': prompt}
                ])

                result = response['message']['content'].strip()

                if "모순 없음" not in result and "no contradiction" not in result.lower():
                    contradictions.append(f"{t1} ↔ {t2}: {result[:150]}")
            except Exception as e:
                # LLM 호출 실패 시 무시
                pass

        return contradictions


class CompletenessChecker:
    """완전성 검사기"""

    @staticmethod
    def check_answer_completeness(query: str, context: str, client) -> ValidationResult:
        """
        질문에 대한 답변 완전성 검사

        Args:
            query: 사용자 질문
            context: 검색된 컨텍스트
            client: Ollama 클라이언트

        Returns:
            ValidationResult: 검증 결과
        """
        prompt = f"""질문: {query}

제공된 컨텍스트:
{context[:2000]}...

이 컨텍스트로 질문에 **완전히** 답할 수 있는지 평가하세요:

1. 완전성 점수 (0-100): 질문에 필요한 정보가 얼마나 포함되어 있는가?
2. 부족한 정보: 무엇이 더 필요한가? (없으면 "없음")
3. 개선 제안: 어떤 추가 정보가 있으면 좋을까? (없으면 "없음")

형식:
완전성: 85
부족: [내용]
제안: [내용]"""

        try:
            response = client.chat(model=client.model, messages=[
                {'role': 'user', 'content': prompt}
            ])

            result = response['message']['content']

            # 파싱
            score_match = re.search(r'완전성[:\s]+(\d+)', result)
            score = int(score_match.group(1)) / 100.0 if score_match else 0.5

            issues = []
            recommendations = []

            for line in result.split('\n'):
                if '부족' in line and '없음' not in line:
                    issues.append(line.split(':', 1)[-1].strip())
                elif '제안' in line and '없음' not in line:
                    recommendations.append(line.split(':', 1)[-1].strip())

            return ValidationResult(
                is_valid=(score >= 0.7),
                confidence=score,
                issues=issues,
                recommendations=recommendations
            )

        except Exception as e:
            # 오류 발생 시 기본값 반환
            return ValidationResult(
                is_valid=False,
                confidence=0.5,
                issues=[f"LLM 호출 실패: {str(e)}"],
                recommendations=[]
            )


class CredibilityAssessor:
    """신뢰성 평가기"""

    @staticmethod
    def assess_source_credibility(sections: List[Tuple[str, str]]) -> Dict[str, float]:
        """
        섹션별 신뢰성 평가 (휴리스틱)

        평가 기준:
        - 내용 길이 (너무 짧으면 낮은 신뢰도)
        - 구체성 (숫자, 날짜, 고유명사 포함 시 높은 신뢰도)
        - 구조화 정도 (리스트, 표 등 포함 시 높은 신뢰도)

        Args:
            sections: [(제목, 내용), ...] 리스트

        Returns:
            Dict[str, float]: {섹션명: 신뢰도(0-1)}
        """
        credibility_scores = {}

        for title, content in sections:
            score = 0.5  # 기본 점수

            # 1. 길이 평가 (100-2000자가 적정)
            length = len(content)
            if 100 <= length <= 2000:
                score += 0.2
            elif length > 2000:
                score += 0.1
            else:
                score -= 0.1

            # 2. 구체성 평가 (숫자, 날짜, 고유명사)
            has_numbers = bool(re.search(r'\d+', content))
            has_dates = bool(re.search(r'\d{4}[-년./]\d{1,2}[-월./]\d{1,2}', content))
            capital_words = len(re.findall(r'[A-Z][a-z]+', content))

            if has_numbers:
                score += 0.1
            if has_dates:
                score += 0.1
            if capital_words > 3:
                score += 0.1

            # 3. 구조화 정도 (리스트, 표)
            has_lists = bool(re.search(r'[\n\r][\s]*[-•\d.]+\s', content))
            has_tables = '|' in content or '\t' in content

            if has_lists:
                score += 0.1
            if has_tables:
                score += 0.1

            # 최종 점수 정규화 (0-1)
            credibility_scores[title] = max(0.0, min(1.0, score))

        return credibility_scores

    @staticmethod
    def assess_information_recency(content: str) -> float:
        """
        정보 최신성 평가

        Args:
            content: 평가할 내용

        Returns:
            float: 최신성 점수 (0-1)
        """
        # 날짜 패턴 검색
        date_patterns = [
            r'20(\d{2})[년\-./]',  # 2020~2099
            r'19(\d{2})[년\-./]',  # 1900~1999
        ]

        years = []
        for pattern in date_patterns:
            matches = re.findall(pattern, content)
            years.extend([int('20' + m) if len(m) == 2 and int(m) <= 99 else int('19' + m)
                          for m in matches])

        if not years:
            # 날짜 정보 없으면 중립 (0.5)
            return 0.5

        # 가장 최근 년도 기준 평가
        max_year = max(years)
        current_year = 2026  # 현재 년도

        # 최근 5년 이내면 1.0, 그 이상은 감소
        age = current_year - max_year
        if age <= 5:
            return 1.0
        elif age <= 10:
            return 0.8
        elif age <= 20:
            return 0.5
        else:
            return 0.3


class QualityMetrics:
    """종합 품질 지표"""

    @staticmethod
    def calculate_overall_quality(section_count: int,
                                  avg_relevance: float,
                                  evidence_verified_ratio: float,
                                  contradiction_count: int,
                                  completeness: float,
                                  avg_credibility: float) -> Dict[str, float]:
        """
        종합 품질 점수 계산

        Args:
            section_count: 검색된 섹션 수
            avg_relevance: 평균 관련성 점수 (0-1)
            evidence_verified_ratio: 증거 검증 비율 (0-1)
            contradiction_count: 모순 개수
            completeness: 완전성 점수 (0-1)
            avg_credibility: 평균 신뢰성 점수 (0-1)

        Returns:
            Dict[str, float]: 각 품질 지표와 종합 점수
        """
        # 가중치
        weights = {
            'coverage': 0.15,      # 검색 범위
            'relevance': 0.25,     # 관련성
            'evidence': 0.20,      # 증거 검증
            'consistency': 0.15,   # 일관성 (모순 없음)
            'completeness': 0.15,  # 완전성
            'credibility': 0.10,   # 신뢰성
        }

        # 각 지표 정규화 (0-1)
        coverage_score = min(section_count / 10.0, 1.0)  # 10개 이상이면 만점
        relevance_score = avg_relevance
        evidence_score = evidence_verified_ratio
        consistency_score = max(0, 1.0 - contradiction_count * 0.2)  # 모순 1개당 -0.2
        completeness_score = completeness
        credibility_score = avg_credibility

        # 가중 합계
        overall_score = (
            coverage_score * weights['coverage'] +
            relevance_score * weights['relevance'] +
            evidence_score * weights['evidence'] +
            consistency_score * weights['consistency'] +
            completeness_score * weights['completeness'] +
            credibility_score * weights['credibility']
        )

        return {
            'overall': overall_score,
            'coverage': coverage_score,
            'relevance': relevance_score,
            'evidence': evidence_score,
            'consistency': consistency_score,
            'completeness': completeness_score,
            'credibility': credibility_score
        }
