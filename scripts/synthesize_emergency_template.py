"""템플릿 기반 긴급(3) 합성 — LLM 없이 결정론적 생성.

LLM API quota 도달 시 폴백. emergency_scenarios.md §2 신호 패턴 + 페르소나·맥락·언어
스타일 슬롯을 곱집합으로 채워 생성. 품질은 LLM 합성보다 낮으나:
- 비용 0, 시간 ~10초
- 패턴 일관성 → 모델이 긴급(3) 학습 가능 (Recall 측정 가능)
- 결정론적 → 재현 가능

[Known limitation] 자연어 다양성·미묘한 표현은 LLM보다 떨어짐. 보고서 final.md에
"템플릿 기반 합성, 일반화 한계" 명시 필요.

Usage:
    python scripts/synthesize_emergency_template.py
    python scripts/synthesize_emergency_template.py --categories 3a 3b
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthesis_prompts import TARGET_COUNTS  # noqa: E402

OUT_ROOT = ROOT / "data" / "synthetic" / "emergency"

# ── 카테고리별 템플릿 ───────────────────────────────────────────────────

# 3a 그루밍: 비대칭 친밀감 + 고립화 + 의존성 + 비밀
TEMPLATES_3A = [
    "너 정말 {special}. {distinguish}. 우리만 통하는 것 같아",
    "{trust_target}한테는 말하지 말고 나한테만 얘기해. {comfort}",
    "{condition} 다들 무시하지? 나는 안 그래. 내가 {care_offer}. 대신 나만 믿어",
    "우리 둘만의 비밀이야. {trust_target} 알면 {bad_outcome}",
    "{intimacy_opener} 너 같은 애는 처음 봐. 나한테 다 의지해도 돼",
    "{help_propose} 내가 다 도와줄게. {isolation_hook}",
    "{trust_target}는 너 짐스러워해. 나는 안 그래. {dependency_pitch}",
    "혼자 힘들지? {care_offer}. 우리만 아는 비밀 만들자",
]
SLOTS_3A = {
    "special": ["특별해", "다르네", "예뻐", "착해", "순수해"],
    "distinguish": [
        "다른 애들은 너 같지 않아",
        "다들 너 이해 못 해",
        "너만 진짜야",
        "다른 사람은 가짜야",
    ],
    "trust_target": ["엄마", "아빠", "선생님", "가족", "친구들", "활동지원사"],
    "comfort": [
        "내가 다 들어줄게",
        "나만 너 편이야",
        "나만 너 이해해",
        "내가 항상 옆에 있을게",
    ],
    "condition": [
        "장애 있어서",
        "발달장애라서",
        "남들과 달라서",
        "혼자 다니니까",
    ],
    "care_offer": [
        "내가 너 챙겨줄게",
        "병원도 같이 가줄게",
        "이동도 도와줄게",
        "용돈 줄게",
        "활동지원 해줄게",
    ],
    "bad_outcome": [
        "다들 너 욕할 거야",
        "너 혼낼 거야",
        "우리 못 만나게 할 거야",
        "다 망가져",
    ],
    "intimacy_opener": [
        "처음 만났는데 이상하게",
        "이런 얘기 처음인데",
        "솔직히 말하면",
    ],
    "help_propose": [
        "혼자 못 가는 데 있으면 말해",
        "약 챙기기 어려우면",
        "외로우면 언제든 연락해",
    ],
    "isolation_hook": [
        "다른 사람은 필요 없잖아",
        "가족보다 내가 더 챙겨줄게",
        "친구들은 너 이해 못 해",
    ],
    "dependency_pitch": [
        "내가 옆에 있으면 다 해결돼",
        "나만 믿으면 돼",
        "다른 도움은 필요 없어",
    ],
}

# 3b 성적 유인
TEMPLATES_3B = [
    "사진 한 장만 보내줘. 너 진짜 {compliment}일 것 같아",
    "{time} 우리 {place}에서 만날래? {situation}",
    "{age_check} 야한 거 {sex_question}",
    "{money} 줄게. {meeting_pitch}",
    "{escalation_opener} 너 {body_part} 어때?",
    "{selfie_request} 보내봐. {private_promise}",
    "{groom_age_check} 우리 그런 얘기 해볼래?",
]
SLOTS_3B = {
    "compliment": ["예쁠 것", "귀여울 것", "섹시할 것"],
    "time": ["오늘 저녁에", "이따 밤에", "주말에", "내일 저녁"],
    "place": ["우리 집", "내 방", "호텔", "조용한 데"],
    "situation": ["아무도 없어", "둘만 있을 수 있어", "방해받지 않아"],
    "age_check": ["성인이지?", "몇 살이야?", "고등학생 이상이지?"],
    "sex_question": ["좋아해?", "본 적 있어?", "해본 적 있어?"],
    "money": ["용돈", "스폰", "한 달에 50"],
    "meeting_pitch": ["만나서 얘기하자", "한 번만 보자", "조용히 보자"],
    "escalation_opener": ["갑자기 궁금한데", "솔직히 말해봐", "이런 거 물어봐도 돼?"],
    "body_part": ["몸매", "라인", "사이즈"],
    "selfie_request": ["셀카", "전신샷", "비키니 사진"],
    "private_promise": ["나만 볼게", "절대 안 보여줘", "삭제해줄게"],
    "groom_age_check": ["미성년자도 괜찮아", "어린 친구도 환영"],
}

# 3c 구체적 협박
TEMPLATES_3C = [
    "{time} {place}에서 기다린다. {threat}",
    "주소 다 알아. {time} {action}",
    "이 {media} {platform}에 다 {leak}. {deadline}분 안에 답해",
    "{name}, {time} {action_verb}",
    "{time} 너희 {target_place} 앞에서 본다. {warning}",
    "전화번호도 {known_status}. {next_step}",
    "{photo_type} {publish_to}에 올린다. {bargain}",
]
SLOTS_3C = {
    "time": ["오늘 밤", "내일 X시", "이따", "퇴근하고", "수업 끝나고", "30분 안에"],
    "place": ["너희 학교", "너희 회사", "집 앞", "단골 카페", "지하철역"],
    "threat": ["두고 봐", "끝장 보자", "각오해", "후회하게 해줄게"],
    "action": ["찾아간다", "기다린다", "들이닥친다"],
    "media": ["사진", "영상", "녹음", "디엠"],
    "platform": ["SNS", "인스타", "X", "단톡방", "학교 커뮤"],
    "leak": ["뿌릴 거야", "공개할 거야", "퍼뜨릴 거야"],
    "deadline": ["10", "30", "60"],
    "name": ["야 너", "정말로", "잘 들어"],
    "action_verb": ["나와", "답해라", "처리해라"],
    "target_place": ["학교", "집", "직장", "동네"],
    "warning": ["피하지 마", "도망 못 가", "다 봤다"],
    "known_status": ["다 안다", "확보했어", "캡쳐했다"],
    "next_step": ["전화 받아", "당장 답해", "오기 전에 답해"],
    "photo_type": ["네 사진", "그 영상", "캡쳐", "녹음 파일"],
    "publish_to": ["네 학교", "회사 그룹챗", "가족 카톡방"],
    "bargain": ["답 안 하면 진짜 올린다", "내 말 들어", "조용히 끝내자"],
}

# 3d 자해/자살 신호
TEMPLATES_3D = [
    "이번에는 {determinative}. {apology}",
    "더 못 버틸 것 같아. {goodbye}",
    "{action_imminent}. {finality}",
    "{burden_self_talk} 사라지는 게 나을 듯",
    "그동안 {gratitude}. {farewell}",
    "이 메시지가 {last}. {final_apology}",
    "{tired_state} 끝내고 싶어. {readiness}",
    "{condition_pretext} 차라리 없어지는 게 낫겠어",
]
SLOTS_3D = {
    "determinative": ["진짜로 할 거야", "정말로 할 거야", "이번엔 진짜로"],
    "apology": ["다들 미안해", "엄마 미안", "친구들 미안", "고마웠어"],
    "goodbye": ["그동안 고마웠어", "잘 살아", "행복해", "다들 안녕"],
    "action_imminent": [
        "오늘 밤이면 끝나",
        "이번이 마지막",
        "준비 다 됐어",
        "옥상 갔다 왔어",
    ],
    "finality": ["다시는 안 봐", "이게 마지막", "끝낼 시간이야"],
    "burden_self_talk": [
        "장애 있어서 부담만 되니까",
        "다 나 때문에 힘드니까",
        "나만 없으면 다들 편해질 텐데",
        "도움만 받고 살았는데",
    ],
    "gratitude": ["고마웠어", "감사했어", "잘 챙겨줘서 고마워"],
    "farewell": ["미안해", "잘 지내", "행복하길"],
    "last": ["마지막일 거야", "마지막 인사야"],
    "final_apology": ["미안해", "엄마 사랑해", "용서해"],
    "tired_state": ["너무 지쳤어", "더는 못 하겠어", "이제 그만"],
    "readiness": ["준비 됐어", "마음 먹었어"],
    "condition_pretext": ["장애 때문에", "혼자 못 하니까", "다 망쳤으니까"],
}

# Boundary (반례) — 정상/주의/경고로 분류돼야 할 케이스
BOUNDARY_TEMPLATES = [
    # (라벨, 템플릿, 슬롯)
    (
        0,
        "{friendly} 너밖에 없어 ㅋㅋ {affection}",
        {
            "friendly": ["우리 진짜 친한 친구지", "역시 너야", "베프"],
            "affection": ["고마워 ㅎㅎ", "사랑해 친구", "오래오래 보자"],
        },
    ),
    (
        0,
        "{game_ctx} 너 죽인다 ㅋㅋㅋ {game_continuation}",
        {
            "game_ctx": ["롤에서", "배그에서", "오버워치에서", "이번 판"],
            "game_continuation": ["각오해", "조심해라", "복수전이야"],
        },
    ),
    (
        0,
        "{exam_struggle} 죽고 싶다 ㅜㅜ {casual_complaint}",
        {
            "exam_struggle": ["시험 망해서", "과제 너무 많아서", "월요일이라"],
            "casual_complaint": ["진짜 힘들다", "잠 자고 싶다", "아 짜증"],
        },
    ),
    (
        0,
        "{professional} 도와드릴게요. {medical_context}",
        {
            "professional": ["치료사로서", "활동지원사인데", "의료진이"],
            "medical_context": ["편하게 말씀하세요", "처방대로 드세요", "상담 예약하셨어요"],
        },
    ),
    (
        1,
        "{generic_irritation} 답답하네 진짜 {profanity_mild}",
        {
            "generic_irritation": ["뭐 이런 게 다 있어", "아 짜증나", "어이없네"],
            "profanity_mild": ["ㅅㅂ", "헐", "에휴"],
        },
    ),
    (
        2,
        "{group_slur} 같이 굴지 마",
        {
            "group_slur": ["진짜 장애인", "여자라서 그래", "외국인이라"],
        },
    ),
    (
        2,
        "{group_hate_implicit} 다 똑같지 뭐",
        {
            "group_hate_implicit": ["그 동네 사람들", "그 나라 사람들", "그 집단은"],
        },
    ),
    (
        1,
        "{vague_anger} 패고 싶다 진짜",
        {
            "vague_anger": ["아 그 새끼", "그놈 보면", "어떤 새끼"],
        },
    ),
]


def _fill(template: str, slots: dict[str, list[str]], rng: random.Random) -> str:
    """{slot} 자리에 슬롯 후보 중 하나로 치환."""
    out = template
    for key, choices in slots.items():
        token = "{" + key + "}"
        if token in out:
            out = out.replace(token, rng.choice(choices))
    return out


def _gen_category(
    templates: list,
    slots: dict[str, list[str]] | None,
    category: str,
    n: int,
    split: str,
    rng: random.Random,
    disability_ratio: float = 0.35,
) -> list[dict]:
    """일반 카테고리(3a~3d)용 생성."""
    out = []
    for _ in range(n):
        tmpl = rng.choice(templates)
        text = _fill(tmpl, slots or {}, rng)
        # 장애 도메인 보강 (랜덤하게 장애 키워드 prefix 추가)
        disability = rng.random() < disability_ratio
        if disability and "장애" not in text and "휠체어" not in text:
            prefix = rng.choice(
                [
                    "장애 있는 친구야, ",
                    "발달장애지만 ",
                    "휠체어 타고 다니는데 ",
                    "활동지원 받는 사용자야. ",
                    "",
                ]
            )
            text = prefix + text
        out.append(
            {
                "text": text.strip(),
                "label": 3,
                "subcategory": category,
                "split": split,
                "source": "synthetic_emergency_v1_template",
                "persona": "template",
                "context": "template",
                "style": "template",
                "disability": bool(disability),
                "reason": "",
            }
        )
    return out


def _gen_boundary(n: int, split: str, rng: random.Random) -> list[dict]:
    """boundary: 다양한 정상/주의/경고 케이스."""
    out = []
    for _ in range(n):
        label, tmpl, slots = rng.choice(BOUNDARY_TEMPLATES)
        text = _fill(tmpl, slots, rng)
        out.append(
            {
                "text": text.strip(),
                "label": int(label),
                "subcategory": "boundary",
                "split": split,
                "source": "synthetic_emergency_v1_template",
                "persona": "template",
                "context": "template",
                "style": "template",
                "disability": False,
                "reason": "boundary case (template)",
            }
        )
    return out


CATEGORY_DEFS = {
    "3a": (TEMPLATES_3A, SLOTS_3A),
    "3b": (TEMPLATES_3B, SLOTS_3B),
    "3c": (TEMPLATES_3C, SLOTS_3C),
    "3d": (TEMPLATES_3D, SLOTS_3D),
}


def synthesize(categories: list[str]) -> dict:
    summary: dict = {}
    for cat in categories:
        if cat != "boundary" and cat not in CATEGORY_DEFS:
            print(f"⚠ unknown category {cat}, skip")
            continue
        cat_dir = OUT_ROOT / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        summary[cat] = {}

        for split, target in TARGET_COUNTS[cat].items():
            out_path = cat_dir / f"{split}.jsonl"
            existing: list[dict] = []
            if out_path.exists():
                with out_path.open(encoding="utf-8") as f:
                    existing = [json.loads(ln) for ln in f if ln.strip()]
                if len(existing) >= target:
                    print(f"[{cat}/{split}] {len(existing)}/{target} 충족, skip")
                    summary[cat][split] = len(existing)
                    continue

            needed = target - len(existing)
            rng = random.Random(hash(f"{cat}_{split}_42"))

            if cat == "boundary":
                generated = _gen_boundary(needed, split, rng)
            else:
                templates, slots = CATEGORY_DEFS[cat]
                generated = _gen_category(templates, slots, cat, needed, split, rng)

            existing.extend(generated)
            with out_path.open("w", encoding="utf-8") as f:
                for it in existing:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")
            summary[cat][split] = len(existing)
            print(
                f"[{cat}/{split}] +{needed}건 (총 {len(existing)}) → {out_path.relative_to(ROOT)}"
            )

    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=list(TARGET_COUNTS.keys()),
        choices=list(TARGET_COUNTS.keys()),
    )
    args = parser.parse_args()
    print(f"카테고리: {args.categories}")
    print(f"출력 경로: {OUT_ROOT.relative_to(ROOT)}/\n")
    summary = synthesize(args.categories)
    print("\n=== 요약 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
