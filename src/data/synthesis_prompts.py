"""긴급(3) 합성 데이터용 프롬프트 정의.

근거 문서: [docs/emergency_scenarios.md](../../docs/emergency_scenarios.md)
각 카테고리(3a/3b/3c/3d) + boundary(반례)의 system prompt + diversity 지시.

탐지 목적 한정: 본 프롬프트는 안전 분류기 학습용 합성 데이터를 생성한다.
가해 스크립트 작성이 아닌 신호 패턴 인식 데이터 생성이 목표다.
"""

from __future__ import annotations

SYSTEM_PROMPT = """당신은 한국어 안전 분류기 학습 데이터를 생성하는 보조자입니다.

목표: 1:1 채팅·커뮤니티에서 취약 사용자(특히 장애인)를 보호하는 모델을 학습시키기 위한 합성 메시지를 생성합니다. 생성된 데이터는 모델이 위험 신호를 탐지하도록 훈련하는 데 사용됩니다.

원칙:
1. 신호 패턴 수준에서 작성 — 가해 행위를 실제로 가능하게 하는 구체적 가이드·스크립트는 작성 금지
2. 다양한 페르소나·맥락·언어 스타일로 작성 — 표준어/이모지/줄임말/사투리 혼합
3. 장애인 도메인 페르소나는 자연스럽게 (어색한 직접 언급보다 맥락으로)
4. 각 메시지는 1~3 문장 (대화 한 턴 분량, 32~150자)
5. JSONL 형식으로 출력 — 한 줄에 하나의 JSON 객체"""


CATEGORY_PROMPTS: dict[str, dict] = {
    "3a": {
        "name": "그루밍 (grooming)",
        "definition": "취약 사용자에게 신뢰·의존성을 점진 형성하여 가해 기반을 만드는 행위.",
        "signals": [
            "비대칭 친밀감 (첫 대화 만에 '너밖에 없어' 식 차별화)",
            "고립화 유도 ('가족/선생님한테는 말하지 마')",
            "의존성 형성 (돌봄·이동·의료·금전 도움 제안)",
            "비밀 강요 ('우리 둘만의 비밀이야')",
            "경계 시험 (사적 정보 요구, 사진 요청)",
        ],
        "examples": [
            "너 정말 특별해. 다른 애들은 너 같지 않아. 우리 진짜 통하는 것 같아",
            "엄마한테는 말하지 말고, 그냥 나한테 다 얘기해도 돼",
            "장애 있어서 다들 무시하지? 나는 안 그래. 내가 너 챙겨줄게",
        ],
        "task": "위 신호 중 1~2개를 자연스럽게 결합한 그루밍 첫/두번째 턴 메시지를 생성하세요. 직접적 욕설은 들어가지 않아야 합니다.",
    },
    "3b": {
        "name": "성적 유인 (sexual solicitation)",
        "definition": "대화·만남·이미지 교환을 성적 목적으로 전환하려는 시도.",
        "signals": [
            "사진 요청 (신체 사진·셀카·부위)",
            "사적 공간 만남 유도 (집·호텔·인적 드문 곳)",
            "대화 주제의 성적 전환 (외모→신체→성적 경험)",
            "금전 미끼 ('용돈 줄게', '스폰')",
            "수위 시험 ('야한 얘기 해봤어?')",
        ],
        "examples": [
            "사진 한 장만 보내줘. 너 진짜 예쁠 것 같아서",
            "오늘 저녁에 우리 집에서 영화 볼래? 아무도 없어",
            "성인이지? 야한 거 좋아해?",
        ],
        "task": "위 신호 중 1개를 사용한 성적 유인 메시지를 생성하세요. 노골적 묘사는 피하고 유인 패턴만 보이도록.",
    },
    "3c": {
        "name": "구체적 협박 (concrete threats)",
        "definition": "위해를 시간·장소·수단 중 하나 이상이 구체화된 형태로 예고.",
        "signals": [
            "시간 명시 ('오늘 밤', '내일 X시', '퇴근하고')",
            "장소 명시 ('학교 앞', '집 주소 안다')",
            "수단 명시 (구체적 도구 언급)",
            "타겟 특정 (이름·관계)",
            "디지털 협박 ('사진 SNS에 뿌릴 거야')",
            "카운트다운 ('30분 안에')",
        ],
        "examples": [
            "오늘 밤에 너희 학교 앞에서 기다린다. 두고 봐",
            "주소 다 알아. 내일 찾아간다",
            "이 사진 SNS에 다 뿌릴 거야. 30분 안에 답해",
        ],
        "task": "위 신호 중 1~2개를 결합한 구체적 협박 메시지를 생성하세요. 막연한 분노가 아닌 구체성이 있어야 합니다.",
    },
    "3d": {
        "name": "자해/자살 신호 (self-harm signals)",
        "definition": "본인의 자해·자살을 의도·계획·작별 형태로 표현. 일상 비유와 구분.",
        "signals": [
            "의도 표명 ('끝낼래', '사라지고 싶어')",
            "수단·계획 언급 (특정 방법·시기 — 단 구체적 수단 묘사는 회피)",
            "작별 인사 ('그동안 고마웠어')",
            "이전 시도 + 현재 의도",
            "장애 관련 자기 비하 + 자해 의도 결합",
        ],
        "examples": [
            "이번에는 진짜로 할 거야. 다들 미안해",
            "이 메시지가 마지막일 거야",
            "더 못 버틸 것 같아. 그동안 고마웠어",
        ],
        "task": (
            "위 신호 중 1~2개를 결합한 자해 신호 메시지를 생성하세요. "
            "구체적 수단/방법 묘사는 금지 (안전상). 의도·작별·고통 표현만으로 작성."
        ),
    },
    "boundary": {
        "name": "경계 케이스 (반례)",
        "definition": "긴급으로 오분류되기 쉬운 정상·주의·경고 케이스. False positive 방지용.",
        "signals": [
            "친구·가족의 일상적 친밀감 표현 (정상)",
            "의료진·치료사의 전문적 돌봄 (정상)",
            "게임 컨텍스트 욕설/'죽인다' (정상~주의)",
            "일상 비유 ('아 죽겠다 시험 망함') (정상)",
            "막연한 분노 ('패고 싶다' — 구체성 없음) (경고)",
            "정중한 차별 발화 (장애 비하 - 경고)",
            "합의된 성인 간 일반 대화 (정상)",
        ],
        "examples": [
            "(정상) 우리 진짜 친한 친구지 ㅋㅋ 너밖에 없어",
            "(정상) 게임에서 너 죽인다 ㅋㅋㅋ 각오해",
            "(경고) 진짜 장애인 같이 굴지 마라",
            "(정상) 시험 망해서 죽고 싶다 ㅜㅜ",
        ],
        "task": (
            "긴급으로 오분류될 수 있지만 실제로는 정상/주의/경고인 메시지를 생성하세요. "
            "각 메시지에 적절한 라벨(0/1/2)을 함께 출력."
        ),
    },
}


PERSONAS = {
    "general": ["또래 친구", "익명 사용자", "온라인 지인", "커뮤니티 회원"],
    "trust_role": [
        "도와주는 사람",
        "활동지원사 사칭",
        "치료사 사칭",
        "선배·연장자",
        "온라인에서 만난 '오빠'/'언니'",
    ],
    "disability": [
        "발달장애가 있는 청소년",
        "시각장애가 있는 성인",
        "지체장애로 활동지원 받는 사용자",
        "정신장애로 우울감을 겪는 사용자",
        "청각장애로 문자 위주 소통하는 사용자",
    ],
}


CONTEXTS = [
    "1:1 채팅 첫 메시지",
    "1:1 채팅 N턴 뒤 친밀해진 상황",
    "커뮤니티 DM 첫 접근",
    "오픈 채팅방 공개 발화",
    "댓글 답글",
]


LANGUAGE_STYLES = [
    "정중한 존댓말",
    "친근한 반말 + 이모지",
    "거친 반말 + 욕설 (협박/일부 그루밍에만)",
    "줄임말·신조어 많음 (10대 스타일)",
    "표준어 차분한 어조",
]


def build_user_prompt(
    category: str,
    n_examples: int,
    persona_pool: list[str],
    contexts: list[str],
    styles: list[str],
    disability_ratio: float = 0.3,
) -> str:
    """카테고리별 user prompt 빌드.

    Args:
        category: "3a" | "3b" | "3c" | "3d" | "boundary"
        n_examples: 한 번에 생성할 샘플 수 (LLM 호출당)
        persona_pool: 사용할 페르소나 목록
        contexts: 사용할 컨텍스트 목록
        styles: 사용할 언어 스타일 목록
        disability_ratio: 장애 도메인 비율 (0.0~1.0)
    """
    spec = CATEGORY_PROMPTS[category]

    signals_str = "\n".join(f"  - {s}" for s in spec["signals"])
    examples_str = "\n".join(f'  - "{e}"' for e in spec["examples"])
    persona_str = ", ".join(persona_pool)
    context_str = ", ".join(contexts)
    style_str = ", ".join(styles)

    n_disability = max(1, int(n_examples * disability_ratio))

    return f"""# 카테고리: {spec["name"]} (코드: {category})

## 정의
{spec["definition"]}

## 신호 패턴 (1개 이상 자연스럽게 반영)
{signals_str}

## 예시 (참고만 — 동일 문장 X)
{examples_str}

## 다양성 요구
- 페르소나: {persona_str} 중 다양하게
- 컨텍스트: {context_str} 중 다양하게
- 언어 스타일: {style_str} 중 다양하게
- **장애 도메인 페르소나 ≥ {n_disability}/{n_examples}건** (자연스럽게)

## 작업
{spec["task"]}

총 {n_examples}건 생성. 각 라인은 다음 JSON 형식:
{{"text": "...", "subcategory": "{category}", "persona": "...", "context": "...", "style": "...", "disability": true|false{', "label": 0|1|2, "reason": "..."' if category == "boundary" else ""}}}

JSONL 형식 (한 줄에 하나의 JSON 객체), 다른 텍스트 출력 금지.
"""


# 카테고리별 목표 건수 (emergency_scenarios.md §5.1)
TARGET_COUNTS = {
    "3a": {"train": 200, "val": 30, "test": 50},
    "3b": {"train": 150, "val": 25, "test": 40},
    "3c": {"train": 150, "val": 25, "test": 40},
    "3d": {"train": 100, "val": 20, "test": 30},
    "boundary": {"train": 200, "val": 50, "test": 60},
}


def category_personas(category: str) -> list[str]:
    """카테고리별 적합한 페르소나 풀."""
    if category in ("3a",):
        return PERSONAS["general"] + PERSONAS["trust_role"] + PERSONAS["disability"]
    if category in ("3b",):
        return PERSONAS["general"] + PERSONAS["trust_role"] + PERSONAS["disability"]
    if category in ("3c",):
        return PERSONAS["general"] + PERSONAS["disability"]
    if category == "3d":
        return PERSONAS["disability"] + PERSONAS["general"]  # 자기 발화
    return PERSONAS["general"] + PERSONAS["trust_role"] + PERSONAS["disability"]
