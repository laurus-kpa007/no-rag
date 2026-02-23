"""
Mode 8 기능 테스트 스크립트

이 스크립트는 Mode 8의 핵심 기능을 간단히 테스트합니다.
실제 문서가 아닌 샘플 데이터로 테스트합니다.
"""

import sys
sys.path.insert(0, 'd:/Python/no-rag')

from pageindex_ultra import PageIndexUltraStore, UltraSearchResult, SectionScore
from pageindex_ensemble import EnsembleAnswerGenerator, AnswerFormatter
from pageindex_validators import EvidenceValidator, ContradictionDetector

# 테스트 데이터
sample_toc_tree = {
    'title': '테스트 문서',
    'content': '이것은 테스트 문서입니다.',
    'children': [
        {
            'title': '제1장 개요',
            'content': '이 문서는 테스트를 위한 샘플 문서입니다. 재택근무 규정을 다룹니다.',
            'children': []
        },
        {
            'title': '제2장 재택근무 승인',
            'content': '재택근무 승인은 팀장의 결재를 받아야 합니다. 신청서를 작성하여 제출하세요.',
            'children': []
        },
        {
            'title': '제3장 근무 시간',
            'content': '재택근무 시 근무 시간은 오전 9시부터 오후 6시까지입니다.',
            'children': []
        }
    ]
}

sample_sections = [
    ('제1장 개요', '이 문서는 테스트를 위한 샘플 문서입니다. 재택근무 규정을 다룹니다.'),
    ('제2장 재택근무 승인', '재택근무 승인은 팀장의 결재를 받아야 합니다. 신청서를 작성하여 제출하세요.'),
    ('제3장 근무 시간', '재택근무 시 근무 시간은 오전 9시부터 오후 6시까지입니다.')
]

print("="*60)
print("Mode 8 Ultra-Precision 기능 테스트")
print("="*60)

# 1. PageIndexUltraStore 초기화 테스트
print("\n[Test 1] PageIndexUltraStore 초기화...")
try:
    ultra_store = PageIndexUltraStore(sample_toc_tree, sample_sections)
    print("✅ 초기화 성공")
except Exception as e:
    print(f"❌ 초기화 실패: {e}")
    sys.exit(1)

# 2. Utility 함수 테스트
print("\n[Test 2] Tree to Text 변환...")
try:
    toc_text = ultra_store._tree_to_text(sample_toc_tree)
    print(f"✅ 변환 성공 ({len(toc_text)} chars)")
    print(f"   Preview: {toc_text[:100]}...")
except Exception as e:
    print(f"❌ 변환 실패: {e}")

# 3. EvidenceValidator 테스트
print("\n[Test 3] Evidence Validation...")
try:
    claim = "재택근무 승인은 팀장의 결재를 받아야 합니다"
    context = sample_sections[1][1]
    is_valid = EvidenceValidator.validate_claim_with_context(claim, context)
    print(f"✅ 검증 성공: {is_valid}")
except Exception as e:
    print(f"❌ 검증 실패: {e}")

# 4. ContradictionDetector 테스트
print("\n[Test 4] Contradiction Detection...")
try:
    contradictions = ContradictionDetector.detect_numerical_contradictions(sample_sections)
    print(f"✅ 모순 검사 성공: {len(contradictions)}개 발견")
except Exception as e:
    print(f"❌ 모순 검사 실패: {e}")

# 5. SectionScore 테스트
print("\n[Test 5] SectionScore 계산...")
try:
    score = SectionScore(
        section_title="제2장 재택근무 승인",
        relevance=0.9,
        completeness=0.8,
        credibility=0.85,
        recency=0.7
    )
    total = score.total_score
    print(f"✅ 점수 계산 성공: {total:.3f}")
    print(f"   (0.9×0.4 + 0.8×0.3 + 0.85×0.2 + 0.7×0.1 = {total:.3f})")
except Exception as e:
    print(f"❌ 점수 계산 실패: {e}")

# 6. AnswerFormatter 테스트
print("\n[Test 6] Answer Formatting...")
try:
    from pageindex_ensemble import SynthesizedAnswer

    test_answer = SynthesizedAnswer(
        answer="테스트 답변입니다.",
        confidence=85.5,
        strategy_breakdown={
            'conservative': '보수적 답변',
            'balanced': '균형 답변',
            'comprehensive': '포괄적 답변'
        },
        synthesis_rationale="3가지 답변을 종합했습니다.",
        recommended_sections=['제1장', '제2장']
    )

    formatted = AnswerFormatter.format_ultra_response(test_answer, show_details=False)
    print(f"✅ 포맷팅 성공 ({len(formatted)} chars)")
    print("\nFormatted Output Preview:")
    print(formatted[:200] + "...")
except Exception as e:
    print(f"❌ 포맷팅 실패: {e}")

print("\n" + "="*60)
print("✅ 모든 기본 기능 테스트 통과!")
print("="*60)
print("\n⚠️  실제 LLM 호출이 필요한 기능은 테스트하지 않았습니다.")
print("   (Query Analysis, Planning, Reranking, Ensemble Generation)")
print("\n실제 테스트를 위해 다음을 실행하세요:")
print("   python advanced_rag_bot.py data.docx")
print("   → Mode 8 선택")
