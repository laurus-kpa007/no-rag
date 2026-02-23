# PageIndex 모드 가이드

## 한 줄 요약

> **벡터 DB 없이**, 문서의 **목차(계층 구조)**를 활용하여 LLM이 **스스로 필요한 섹션을 찾아가는** 검색 방식입니다.

---

## 기존 RAG vs PageIndex 비교

```mermaid
graph LR
    subgraph 기존 RAG ["기존 RAG (Mode 2~5)"]
        A1[문서] --> A2[청크로 잘라냄]
        A2 --> A3[벡터 임베딩 생성]
        A3 --> A4[벡터 DB 저장]
        A4 --> A5["유사도 검색<br/>(코사인 유사도)"]
        A5 --> A6[상위 K개 반환]
    end

    subgraph PageIndex ["PageIndex (Mode 6)"]
        B1[문서] --> B2["계층 트리로 파싱<br/>(제목/본문/표)"]
        B2 --> B3[각 노드 요약 생성]
        B3 --> B4["목차 구성<br/>(LLM이 읽을 수 있는 형태)"]
        B4 --> B5["LLM이 목차를 보고<br/>직접 섹션 선택"]
        B5 --> B6["선택 섹션 내용 확인<br/>+ 추가 탐색 판단"]
    end

    style 기존 RAG fill:#ffeedd,stroke:#cc8800
    style PageIndex fill:#ddffdd,stroke:#00aa44
```

### 핵심 차이점

| 구분 | 기존 RAG | PageIndex |
|------|----------|-----------|
| **검색 기준** | 벡터 유사도 (수학적 거리) | 문서의 논리적 위치 (목차 구조) |
| **필요 인프라** | 벡터 DB (ChromaDB 등) | 없음 (트리 구조만 사용) |
| **문서 분할** | 고정 크기 청크 (500자) | 제목(Heading) 기반 자연스러운 분할 |
| **검색 주체** | 알고리즘 (코사인 유사도) | LLM이 직접 추론하여 선택 |
| **출처 표시** | 청크 번호 | 섹션 제목 + 근거 문장 인용 |

---

## 전체 동작 흐름

```mermaid
flowchart TD
    START([사용자가 Mode 6 선택]) --> CHECK{PageIndex<br/>이미 구축됨?}

    CHECK -->|No| BUILD["🔨 인덱싱 단계<br/>(최초 1회만 실행)"]
    CHECK -->|Yes, 캐시 로드| QUERY

    BUILD --> STEP1["1단계: 문서 → 트리 파싱<br/>(Heading 기반 계층 구조)"]
    STEP1 --> STEP2["2단계: 각 노드 LLM 요약 생성<br/>(Bottom-up 방식)"]
    STEP2 --> STEP3["3단계: 전체 목차 구성<br/>([노드ID] 제목 — 요약)"]
    STEP3 --> CACHE["💾 캐시 저장<br/>(다음 실행시 재사용)"]
    CACHE --> QUERY

    QUERY([사용자 질문 입력]) --> PLANNING

    subgraph AGENT ["에이전틱 추론 탐색"]
        PLANNING["📋 1단계: Planning<br/>LLM이 목차를 훑어보고<br/>관련 섹션 ID 선택"]
        PLANNING --> INSPECT["🔍 2단계: Inspection<br/>선택된 섹션 내용 읽기"]
        INSPECT --> JUDGE{충분한<br/>정보인가?}
        JUDGE -->|Yes| COLLECT["✅ 섹션 수집 완료"]
        JUDGE -->|No| NEXT["다른 섹션으로 이동<br/>(목차에서 추가 선택)"]
        NEXT --> INSPECT
    end

    COLLECT --> ANSWER["💬 답변 생성<br/>[섹션 제목] + [근거 문장] 포함"]

    style BUILD fill:#fff3cd,stroke:#ffc107
    style AGENT fill:#e8f5e9,stroke:#4caf50
    style ANSWER fill:#e3f2fd,stroke:#2196f3
```

---

## 1단계: 문서 → 트리 파싱

`.docx` 파일의 Heading 스타일을 기준으로 문서를 트리 구조로 변환합니다.

### 변환 예시

```mermaid
graph TD
    subgraph 원본문서 ["📄 원본 .docx 문서"]
        D1["Heading 1: 제1장 총칙"]
        D2["  본문: 이 규정은..."]
        D3["  Heading 2: 제1조 목적"]
        D4["    본문: 본 규정의 목적은..."]
        D5["  Heading 2: 제2조 적용범위"]
        D6["    본문: 이 규정은 전 직원에..."]
        D7["Heading 1: 제2장 근무"]
        D8["  Heading 2: 제3조 근무시간"]
        D9["    본문: 근무시간은 09:00~18:00..."]
        D10["    [표] 부서별 근무시간표"]
    end

    subgraph 트리 ["🌳 변환된 트리 구조"]
        R["[000] 문서 전체<br/>(루트)"]
        N1["[001] 제1장 총칙<br/>본문: 이 규정은..."]
        N2["[002] 제1조 목적<br/>본문: 본 규정의 목적은..."]
        N3["[003] 제2조 적용범위<br/>본문: 이 규정은 전 직원에..."]
        N4["[004] 제2장 근무"]
        N5["[005] 제3조 근무시간<br/>본문: 근무시간은 09:00~18:00...<br/>[표] 부서별 근무시간표"]

        R --> N1
        R --> N4
        N1 --> N2
        N1 --> N3
        N4 --> N5
    end

    style 원본문서 fill:#fff8e1,stroke:#ff8f00
    style 트리 fill:#e8f5e9,stroke:#388e3c
```

### 지원하는 구조 패턴

| 문서 형식 | 인식되는 헤딩 |
|-----------|-------------|
| `.docx` | `Heading 1`, `Heading 2`, ... `Heading 6`, 한국어 `제목` 스타일 |
| `.md` | `#`, `##`, `###`, ... `######` |
| `.txt` / `.md` | `제N장`, `제N절`, `제N조` (한국어 규정 패턴) |
| 헤딩 없는 문서 | 자동으로 2000자 단위 분할 (폴백) |

---

## 2단계: 노드 요약 생성

트리의 **리프(말단) 노드부터 상향식(Bottom-up)**으로 LLM이 각 노드를 요약합니다.

```mermaid
graph BT
    subgraph 요약순서 ["요약 생성 순서 (Bottom-up)"]
        L1["[002] 제1조 목적<br/>요약: '본 규정의 목적과<br/>적용 대상을 정의'"]
        L2["[003] 제2조 적용범위<br/>요약: '전 직원 대상,<br/>계약직 포함'"]
        L3["[005] 제3조 근무시간<br/>요약: '09:00~18:00 기본근무,<br/>부서별 탄력근무제 가능'"]

        P1["[001] 제1장 총칙<br/>요약: '규정 목적 및 적용범위 정의,<br/>하위: 목적, 적용범위'"]
        P2["[004] 제2장 근무<br/>요약: '근무시간 및 탄력근무제<br/>관련 규정'"]

        L1 -->|"③ 부모 요약"| P1
        L2 -->|"③ 부모 요약"| P1
        L3 -->|"③ 부모 요약"| P2

        R["[000] 문서 전체<br/>요약: '2개 장으로 구성:<br/>총칙, 근무 규정'"]
        P1 -->|"④ 루트 요약"| R
        P2 -->|"④ 루트 요약"| R
    end

    L1 -.-|"① 먼저"| L1
    L2 -.-|"① 먼저"| L2
    L3 -.-|"② 그 다음"| L3

    style 요약순서 fill:#f3e5f5,stroke:#7b1fa2
```

### 생성되는 목차 형태 (LLM이 읽는 형태)

```
[001] 제1장 총칙 — 규정 목적 및 적용범위 정의, 전 직원 대상
  [002] 제1조 목적 — 본 규정의 목적과 적용 대상을 정의
  [003] 제2조 적용범위 — 전 직원 대상, 계약직 포함
[004] 제2장 근무 — 근무시간 및 탄력근무제 관련 규정
  [005] 제3조 근무시간 — 09:00~18:00 기본근무, 부서별 탄력근무제 가능
```

---

## 3단계: 에이전틱 추론 탐색

사용자 질문이 들어오면 LLM이 **감사관(Auditor) 페르소나**로 목차를 보고 직접 정보를 찾아갑니다.

### Planning → Inspection 루프

```mermaid
sequenceDiagram
    participant U as 사용자
    participant S as 시스템
    participant LLM as LLM (감사관)

    U->>S: "근무시간이 어떻게 되나요?"
    S->>LLM: [Planning] 목차 전체 + 질문 전달

    Note over LLM: 목차를 훑어본다...<br/>"근무시간은 제2장 > 제3조에<br/>있을 것 같다"

    LLM->>S: 선택: [005]<br/>이유: 근무시간 관련 조항

    S->>S: 노드 [005]의 전체 내용 로드
    S->>LLM: [Inspection] 섹션 내용 + 질문

    Note over LLM: 내용을 읽는다...<br/>"09:00~18:00, 탄력근무제 정보가<br/>충분하다"

    LLM->>S: 충분: Yes<br/>근거문장: "근무시간은 09:00~18:00으로 한다"

    S->>LLM: [답변 생성] 수집된 섹션 기반

    LLM->>U: 📋 답변 + [섹션 제목] + [근거 문장]
```

### 정보가 부족할 때 (Loop 발생)

```mermaid
sequenceDiagram
    participant U as 사용자
    participant S as 시스템
    participant LLM as LLM (감사관)

    U->>S: "연차와 병가의 차이점은?"
    S->>LLM: [Planning] 목차 + 질문

    LLM->>S: 선택: [010] (제5조 연차휴가)

    S->>LLM: [Inspection #1] 연차휴가 섹션 내용

    Note over LLM: "연차 정보는 있지만<br/>병가 정보가 없다..."

    LLM->>S: 충분: No<br/>근거문장: "연차는 연 15일 부여"<br/>추가탐색: [012]

    S->>LLM: [Inspection #2] 병가 섹션 내용

    Note over LLM: "이제 둘 다 있다!"

    LLM->>S: 충분: Yes<br/>근거문장: "병가는 연 60일 이내"

    S->>LLM: [답변 생성] 2개 섹션 기반

    LLM->>U: 📋 비교 답변 +<br/>[제5조 연차휴가] 근거 +<br/>[제7조 병가] 근거
```

---

## 답변 출력 형태

PageIndex 모드의 답변은 항상 **출처(섹션 제목)**와 **근거 문장**을 포함합니다.

```
📋 답변 예시:

근무시간은 오전 9시부터 오후 6시까지입니다.

**[근거]**
- **섹션:** 제2장 근무 > 제3조 근무시간
- **원문 인용:** "근무시간은 09:00부터 18:00까지로 하며,
  점심시간은 12:00~13:00으로 한다."

단, 부서별로 탄력근무제를 적용할 수 있습니다.

- **섹션:** 제2장 근무 > 제4조 탄력근무
- **원문 인용:** "부서장 승인 하에 출퇴근 시간을
  1시간 범위 내에서 조정할 수 있다."
```

---

## 설정값

`Config` 클래스에서 조정 가능한 PageIndex 관련 설정:

| 설정 | 기본값 | 설명 |
|------|--------|------|
| `PAGEINDEX_MAX_INSPECT_LOOPS` | `3` | Inspection 단계 최대 반복 횟수 |
| `PAGEINDEX_SUMMARY_MAX_CHARS` | `200` | 각 노드 요약의 최대 글자 수 |

---

## 캐시 동작

```mermaid
flowchart LR
    subgraph 최초실행 ["최초 실행"]
        A1["문서 파싱<br/>(트리 구축)"] --> A2["노드 요약 생성<br/>(LLM 호출 N회)"]
        A2 --> A3["💾 .rag_cache/pageindex.pkl<br/>저장"]
    end

    subgraph 재실행 ["2회차 이후"]
        B1["💾 캐시 파일 확인"] --> B2["트리 + 요약<br/>즉시 로드"]
        B2 --> B3["바로 질의 가능"]
    end

    최초실행 -.->|"다음 실행시"| 재실행

    style 최초실행 fill:#fff3cd,stroke:#ffc107
    style 재실행 fill:#c8e6c9,stroke:#43a047
```

- 최초 실행 시에만 LLM 호출이 발생합니다 (노드 수에 비례)
- 이후에는 `.rag_cache/pageindex.pkl`에서 즉시 로드됩니다
- 문서가 변경되면 기존 캐시를 삭제하고 재구축하면 됩니다

---

## 언제 PageIndex를 사용하면 좋은가?

```mermaid
graph TD
    Q1{문서에 목차/제목<br/>구조가 있는가?}
    Q1 -->|Yes| Q2{규정, 매뉴얼,<br/>계약서 등인가?}
    Q1 -->|No| REC1["Mode 4 또는 5 추천<br/>(하이브리드/자동)"]

    Q2 -->|Yes| REC2["✅ Mode 6 PageIndex 추천<br/>계층 구조 활용 극대화"]
    Q2 -->|No| Q3{정확한 출처/근거가<br/>중요한가?}

    Q3 -->|Yes| REC2
    Q3 -->|No| REC3["Mode 5 자동모드 추천<br/>(범용적)"]

    style REC2 fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    style REC1 fill:#fff9c4,stroke:#f9a825
    style REC3 fill:#fff9c4,stroke:#f9a825
```

### 적합한 문서 유형
- 사내 규정/취업규칙
- 법률/조례/시행령
- 제품 매뉴얼/가이드
- 계약서/약관
- 학술 논문 (챕터 구조)

### 부적합한 문서 유형
- 구조 없는 자유 형식 메모
- 단순 데이터 목록
- 이미지 중심 문서
