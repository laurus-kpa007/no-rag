"""
PageIndex Ultra-Precision Ensemble Answer Generation Module

이 모듈은 여러 접근법으로 답변을 생성하고 이를 종합하는 앙상블 방식을 구현합니다.

앙상블 전략:
1. Conservative (보수적): 명확히 증명된 정보만 포함
2. Balanced (균형): 직접 증거 + 추론 가능한 정보 포함
3. Comprehensive (포괄적): 관련 가능성이 있는 모든 정보 포함

→ 3가지 답변을 LLM이 비교 분석하여 최종 종합 답변 생성
"""

from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class AnswerStrategy(Enum):
    """답변 생성 전략"""
    CONSERVATIVE = "conservative"      # 보수적: 명확한 증거만
    BALANCED = "balanced"              # 균형: 직접 증거 + 추론
    COMPREHENSIVE = "comprehensive"    # 포괄적: 모든 관련 정보


@dataclass
class EnsembleAnswer:
    """앙상블 답변"""
    strategy: AnswerStrategy
    answer: str
    confidence: float  # 0-1
    sources: List[str]  # 사용된 섹션 제목


@dataclass
class SynthesizedAnswer:
    """종합된 최종 답변"""
    answer: str
    confidence: float  # 0-100
    strategy_breakdown: Dict[str, str]  # {전략명: 답변}
    synthesis_rationale: str  # 종합 근거
    recommended_sections: List[str]  # 참조 섹션


class EnsembleAnswerGenerator:
    """앙상블 답변 생성기"""

    def __init__(self, client, model: str = 'gemma3:12b'):
        """
        Args:
            client: Ollama 클라이언트
            model: 사용할 LLM 모델
        """
        self.client = client
        self.model = model

    def generate_ensemble_answers(self,
                                  query: str,
                                  sections: List[Tuple[str, str]],
                                  section_scores: List[Dict],
                                  evidence_verified: Dict[str, bool] = None) -> List[EnsembleAnswer]:
        """
        3가지 전략으로 답변 생성

        Args:
            query: 사용자 질문
            sections: [(제목, 내용), ...] 검색된 섹션들
            section_scores: 섹션별 점수 정보
            evidence_verified: {섹션명: 검증여부} (옵션)

        Returns:
            List[EnsembleAnswer]: 3가지 전략의 답변들
        """
        answers = []

        # 1. Conservative (보수적) 답변
        conservative_answer = self._generate_conservative(
            query, sections, section_scores, evidence_verified)
        answers.append(conservative_answer)

        # 2. Balanced (균형) 답변
        balanced_answer = self._generate_balanced(
            query, sections, section_scores)
        answers.append(balanced_answer)

        # 3. Comprehensive (포괄적) 답변
        comprehensive_answer = self._generate_comprehensive(
            query, sections, section_scores)
        answers.append(comprehensive_answer)

        return answers

    def _generate_conservative(self,
                               query: str,
                               sections: List[Tuple[str, str]],
                               section_scores: List[Dict],
                               evidence_verified: Dict[str, bool] = None) -> EnsembleAnswer:
        """
        보수적 답변 생성: 명확히 검증된 정보만 사용

        전략:
        - 점수가 높은 상위 30% 섹션만 사용
        - 증거가 검증된 정보만 포함
        - "확실하지 않음" 명시
        """
        # 상위 30% 섹션 선택
        sorted_sections = sorted(
            zip(sections, section_scores),
            key=lambda x: x[1].get('total_score', 0),
            reverse=True
        )
        top_count = max(1, int(len(sorted_sections) * 0.3))
        top_sections = [s for s, _ in sorted_sections[:top_count]]

        # 검증된 섹션만 필터링 (있으면)
        if evidence_verified:
            top_sections = [(t, c) for t, c in top_sections
                            if evidence_verified.get(t, False)]

        # 컨텍스트 구성 (짧게)
        context = "\n\n".join([
            f"[{title}]\n{content[:400]}..."
            for title, content in top_sections
        ])

        prompt = f"""질문: {query}

컨텍스트 (검증된 정보만):
{context}

**보수적 답변 규칙**:
1. 컨텍스트에서 명확히 확인된 정보만 사용
2. 추론이나 가정을 피함
3. 불확실한 부분은 "확인되지 않음" 명시
4. 간결하게 답변 (3-5문장)

답변:"""

        response = self.client.chat(model=self.model, messages=[
            {'role': 'user', 'content': prompt}
        ])

        answer_text = response['message']['content']
        used_sections = [title for title, _ in top_sections]

        return EnsembleAnswer(
            strategy=AnswerStrategy.CONSERVATIVE,
            answer=answer_text,
            confidence=0.9,  # 보수적이므로 높은 신뢰도
            sources=used_sections
        )

    def _generate_balanced(self,
                           query: str,
                           sections: List[Tuple[str, str]],
                           section_scores: List[Dict]) -> EnsembleAnswer:
        """
        균형 답변 생성: 직접 증거 + 합리적 추론

        전략:
        - 상위 50% 섹션 사용
        - 직접 증거를 우선하되, 논리적 추론도 포함
        - 추론 부분은 명시
        """
        # 상위 50% 섹션 선택
        sorted_sections = sorted(
            zip(sections, section_scores),
            key=lambda x: x[1].get('total_score', 0),
            reverse=True
        )
        top_count = max(2, int(len(sorted_sections) * 0.5))
        top_sections = [s for s, _ in sorted_sections[:top_count]]

        # 컨텍스트 구성 (중간 길이)
        context = "\n\n".join([
            f"[{title}]\n{content[:600]}..."
            for title, content in top_sections
        ])

        prompt = f"""질문: {query}

컨텍스트:
{context}

**균형 답변 규칙**:
1. 컨텍스트의 직접 증거를 우선 사용
2. 논리적 추론이 필요한 경우, "추론: ..."으로 명시
3. 직접 증거와 추론을 구분
4. 적절한 길이로 답변 (5-8문장)

답변:"""

        response = self.client.chat(model=self.model, messages=[
            {'role': 'user', 'content': prompt}
        ])

        answer_text = response['message']['content']
        used_sections = [title for title, _ in top_sections]

        return EnsembleAnswer(
            strategy=AnswerStrategy.BALANCED,
            answer=answer_text,
            confidence=0.75,  # 중간 신뢰도
            sources=used_sections
        )

    def _generate_comprehensive(self,
                                query: str,
                                sections: List[Tuple[str, str]],
                                section_scores: List[Dict]) -> EnsembleAnswer:
        """
        포괄적 답변 생성: 관련 가능한 모든 정보 포함

        전략:
        - 모든 섹션 사용 (또는 상위 80%)
        - 관련성이 낮더라도 잠재적 가치가 있으면 포함
        - 배경 정보, 관련 컨텍스트 추가
        """
        # 상위 80% 섹션 선택
        sorted_sections = sorted(
            zip(sections, section_scores),
            key=lambda x: x[1].get('total_score', 0),
            reverse=True
        )
        top_count = max(3, int(len(sorted_sections) * 0.8))
        top_sections = [s for s, _ in sorted_sections[:top_count]]

        # 컨텍스트 구성 (긴 길이)
        context = "\n\n".join([
            f"[{title}]\n{content[:800]}..."
            for title, content in top_sections
        ])

        prompt = f"""질문: {query}

컨텍스트 (전체):
{context}

**포괄적 답변 규칙**:
1. 모든 관련 정보를 포함
2. 배경 지식, 관련 컨텍스트 추가
3. 다양한 관점 제시
4. 상세하게 답변 (8-12문장)
5. 관련 가능성이 있는 정보도 포함

답변:"""

        response = self.client.chat(model=self.model, messages=[
            {'role': 'user', 'content': prompt}
        ])

        answer_text = response['message']['content']
        used_sections = [title for title, _ in top_sections]

        return EnsembleAnswer(
            strategy=AnswerStrategy.COMPREHENSIVE,
            answer=answer_text,
            confidence=0.6,  # 낮은 신뢰도 (추측 포함)
            sources=used_sections
        )

    def synthesize_answers(self,
                          query: str,
                          ensemble_answers: List[EnsembleAnswer],
                          contradictions: List[str] = None) -> SynthesizedAnswer:
        """
        3가지 답변을 비교 분석하여 최종 종합 답변 생성

        Args:
            query: 원래 질문
            ensemble_answers: 3가지 전략의 답변들
            contradictions: 발견된 모순 (옵션)

        Returns:
            SynthesizedAnswer: 종합된 최종 답변
        """
        # 각 전략별 답변 정리
        strategy_breakdown = {}
        for ens_ans in ensemble_answers:
            strategy_name = ens_ans.strategy.value
            strategy_breakdown[strategy_name] = ens_ans.answer

        # 종합 프롬프트 구성
        answers_text = "\n\n".join([
            f"[{ens.strategy.value.upper()} - 신뢰도 {ens.confidence:.0%}]\n{ens.answer}"
            for ens in ensemble_answers
        ])

        contradiction_text = ""
        if contradictions:
            contradiction_text = f"\n\n⚠️ 발견된 모순:\n" + "\n".join(contradictions)

        prompt = f"""질문: {query}

다음은 3가지 다른 전략으로 생성된 답변들입니다:

{answers_text}
{contradiction_text}

**종합 답변 생성 규칙**:
1. 3가지 답변을 비교 분석
2. 공통된 핵심 정보를 추출
3. 상충되는 부분은 신뢰도가 높은 쪽 선택
4. 모순이 있으면 명시
5. 최종 종합 답변 작성 (7-10문장)

종합 답변:"""

        response = self.client.chat(model=self.model, messages=[
            {'role': 'user', 'content': prompt}
        ])

        final_answer = response['message']['content']

        # 종합 근거 생성
        rationale_prompt = f"""다음 종합 답변이 만들어진 근거를 설명하세요:

종합 답변:
{final_answer}

원본 3가지 답변:
{answers_text}

**설명 규칙**:
1. 어떤 부분이 공통되어 채택되었는지
2. 어떤 부분이 충돌하여 조정되었는지
3. 최종 선택의 근거
(3-5문장으로 간단히)

근거:"""

        rationale_response = self.client.chat(model=self.model, messages=[
            {'role': 'user', 'content': rationale_prompt}
        ])

        rationale = rationale_response['message']['content']

        # 신뢰도 계산 (앙상블 답변들의 가중 평균)
        total_confidence = sum(ens.confidence for ens in ensemble_answers) / len(ensemble_answers)
        # 모순이 있으면 신뢰도 감소
        if contradictions:
            total_confidence *= (1 - len(contradictions) * 0.1)

        final_confidence = max(0, min(100, total_confidence * 100))

        # 모든 섹션 수집
        all_sections = []
        for ens in ensemble_answers:
            all_sections.extend(ens.sources)
        recommended_sections = list(set(all_sections))

        return SynthesizedAnswer(
            answer=final_answer,
            confidence=final_confidence,
            strategy_breakdown=strategy_breakdown,
            synthesis_rationale=rationale,
            recommended_sections=recommended_sections
        )


class AnswerFormatter:
    """답변 포맷터"""

    @staticmethod
    def format_ultra_response(synthesized: SynthesizedAnswer,
                              show_details: bool = False) -> str:
        """
        Ultra-Precision 답변을 사용자 친화적으로 포맷팅

        Args:
            synthesized: 종합 답변
            show_details: 상세 정보 표시 여부

        Returns:
            str: 포맷팅된 답변
        """
        lines = []

        # 1. 최종 답변
        lines.append("=" * 60)
        lines.append("📌 Ultra-Precision 답변")
        lines.append("=" * 60)
        lines.append("")
        lines.append(synthesized.answer)
        lines.append("")

        # 2. 신뢰도
        confidence_bar = "█" * int(synthesized.confidence / 5) + "░" * (20 - int(synthesized.confidence / 5))
        lines.append(f"🎯 신뢰도: {synthesized.confidence:.1f}% [{confidence_bar}]")
        lines.append("")

        # 3. 참조 섹션
        lines.append("📚 참조 섹션:")
        for i, section in enumerate(synthesized.recommended_sections, 1):
            lines.append(f"  {i}. {section}")
        lines.append("")

        # 4. 상세 정보 (옵션)
        if show_details:
            lines.append("-" * 60)
            lines.append("🔍 종합 분석 과정")
            lines.append("-" * 60)
            lines.append("")

            # 종합 근거
            lines.append("📊 종합 근거:")
            lines.append(synthesized.synthesis_rationale)
            lines.append("")

            # 전략별 답변
            lines.append("📋 전략별 답변:")
            for strategy, answer in synthesized.strategy_breakdown.items():
                lines.append(f"\n[{strategy.upper()}]")
                lines.append(answer[:300] + "..." if len(answer) > 300 else answer)
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def format_reasoning_trace(trace: List[str]) -> str:
        """
        추론 과정 포맷팅

        Args:
            trace: 추론 과정 로그

        Returns:
            str: 포맷팅된 추론 과정
        """
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("🔬 Ultra-Precision 추론 과정")
        lines.append("=" * 60)

        for i, step in enumerate(trace, 1):
            if step.startswith("Stage"):
                lines.append(f"\n{step}")
            else:
                lines.append(step)

        lines.append("=" * 60)

        return "\n".join(lines)
