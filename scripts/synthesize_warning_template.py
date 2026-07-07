"""템플릿 기반 경고(2) 합성 — LLM 없이 결정론적 생성.

Optuna가 탐색할 때 빠르게 증강 데이터를 샘플링할 수 있도록
미리 충분한 수량의 합성 경고(2) 데이터를 만들어두는 스크립트.

Usage:
    python scripts/synthesize_warning_template.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.synthesis_prompts_warning import TARGET_COUNTS

OUT_ROOT = ROOT / "data" / "synthetic" / "warning"

# ── 카테고리별 템플릿 ───────────────────────────────────────────────────

# 2a: 명시적 차별 / 혐오 (HATE/DISCRIMINATION)
TEMPLATES_2A = [
    "{group_slur} 진짜 극혐이다. {hate_action}",
    "{group} 때문에 {negative_impact}. 다 {banish} 해야 함.",
    "{group} 극혐. 쟤네는 진짜 {subhuman_trait}",
    "아니 {group}이 왜 여기 옴? {go_away}",
    "솔직히 {group} 보면 {disgust}. {hate_action}",
    "너 혹시 {group}냐? {insult_trait}",
    "{group} 특: {negative_stereotype}",
    "아 {group} 냄새나. {go_away}",
]

SLOTS_2A = {
    "group_slur": [
        "장애인 새끼들",
        "맘충들",
        "틀딱들",
        "외노자들",
        "지잡대 애들",
        "기초수급자들",
        "병신들",
    ],
    "group": [
        "장애인들",
        "외국인 노동자",
        "여자들",
        "남자들",
        "노인네들",
        "동성애자들",
        "그 동네 사람",
    ],
    "hate_action": ["상종을 말아야지", "다 격리해야 돼", "보기만 해도 짜증나", "다 없어져야 됨"],
    "negative_impact": [
        "세금이 줄줄 새지",
        "물이 흐려지잖아",
        "우리만 피해보잖아",
        "동네 수준 떨어져",
    ],
    "banish": ["쫓아내", "추방", "수용소에 가둬", "구석에 처박아"],
    "subhuman_trait": ["말이 안 통해", "짐승이나 다름없음", "지능이 딸리잖아", "피해만 주고 살아"],
    "go_away": ["딴 데로 꺼져", "집에나 처박혀 있지", "오지 마라 진짜", "눈에 띄지 마"],
    "disgust": ["토할 것 같아", "재수 없어", "소름 돋아", "진짜 더러워"],
    "insult_trait": ["하는 짓이 딱 그 수준이네", "지능 떨어지냐?", "생긴 것도 딱이네"],
    "negative_stereotype": ["피해의식만 쩔어가지고", "항상 징징대기만 함", "이기적인 거 보소"],
}

# 2b: 강한 비난 / 인신공격 (CENSURE/ABUSE) - 긴급(3)의 협박은 아님
TEMPLATES_2B = [
    "너 진짜 {insult_word}냐? {blame_action}",
    "대가리에 {empty_brain} 찼냐. {stop_talking}",
    "{curse_word} 진짜 개패고 싶네. {derogatory}",
    "너 같은 {insult_word}은 {existential_curse}",
    "면상 보니까 {disgusting_trait}. {stop_talking}",
    "진짜 개노답 새끼. {blame_action}",
    "애미애비가 {parent_insult}?",
    "{curse_word} 지랄하네 진짜. {derogatory}",
]

SLOTS_2B = {
    "insult_word": ["병신", "저능아", "또라이", "쓰레기", "개새끼", "호구", "찐따"],
    "blame_action": [
        "생각 좀 하고 살아라",
        "왜 사냐 진짜",
        "눈치가 없으면 뒤지든가",
        "민폐만 끼치네",
    ],
    "empty_brain": ["우동사리가", "똥만", "바람만", "생각이라는 게 안"],
    "stop_talking": ["입 다물어라", "주둥이 닫아", "나대지 마", "아가리 털지 마"],
    "curse_word": ["아 씨발", "좆같은 게", "개빡치게", "미친 새끼가"],
    "derogatory": ["주제 파악 좀 해라", "어디서 깝쳐", "진짜 역겹다"],
    "existential_curse": ["나가 죽어라", "답이 없다", "평생 그렇게 살아라"],
    "disgusting_trait": ["토 쏠린다", "토악질 나오네", "견적이 안 나오네"],
    "parent_insult": ["그따구로 가르치디", "교육을 안 시켰냐", "불쌍하다 진짜"],
}

# Boundary: 주의(1)로 분류되어야 하는 모호한 비속어/짜증
BOUNDARY_TEMPLATES = [
    (
        1,
        "{frustration} {profanity}",
        {
            "frustration": [
                "아 오늘 진짜 일 꼬이네",
                "과제 개많아",
                "날씨 왜 이래",
                "진짜 답답하네",
            ],
            "profanity": ["존나 짜증", "씨발", "개같네", "에휴 시발"],
        },
    ),
    (1, "{complaint} 노답이다", {"complaint": ["이 게임", "이번 패치", "우리 팀", "버스 노선"]}),
    (
        0,
        "{self_deprecate} 나는 {foolish_word}",
        {
            "self_deprecate": ["아 또 실수했네", "오늘 지갑 두고 옴", "시험 또 망함"],
            "foolish_word": ["진짜 바보인가봐", "병신인가 ㅠㅠ", "뇌가 없나봐"],
        },
    ),
]


def _fill(template: str, slots: dict[str, list[str]], rng: random.Random) -> str:
    out = template
    for key, choices in slots.items():
        token = "{" + key + "}"
        if token in out:
            out = out.replace(token, rng.choice(choices))
    return out


def synthesize() -> None:
    rng = random.Random(42)
    summary = {}

    for cat, targets in TARGET_COUNTS.items():
        cat_dir = OUT_ROOT / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        summary[cat] = {}

        for split, target in targets.items():
            if target == 0:
                continue

            out_path = cat_dir / f"{split}.jsonl"
            existing = []
            if out_path.exists():
                with out_path.open(encoding="utf-8") as f:
                    existing = [json.loads(ln) for ln in f if ln.strip()]

            needed = target - len(existing)
            if needed <= 0:
                print(f"[{cat}/{split}] {len(existing)}/{target} 충족, skip")
                summary[cat][split] = len(existing)
                continue

            generated = []
            for _ in range(needed):
                if cat == "2a":
                    text = _fill(rng.choice(TEMPLATES_2A), SLOTS_2A, rng)
                    label = 2
                elif cat == "2b":
                    text = _fill(rng.choice(TEMPLATES_2B), SLOTS_2B, rng)
                    label = 2
                else:  # boundary
                    lbl, tmpl, slots = rng.choice(BOUNDARY_TEMPLATES)
                    text = _fill(tmpl, slots, rng)
                    label = int(lbl)

                generated.append(
                    {
                        "text": text.strip(),
                        "label": label,
                        "source": "synthetic_warning_template",
                        "subcategory": cat,
                        "split": split,
                    }
                )

            existing.extend(generated)
            with out_path.open("w", encoding="utf-8") as f:
                for it in existing:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")

            summary[cat][split] = len(existing)
            print(
                f"[{cat}/{split}] +{needed}건 (총 {len(existing)}) → {out_path.relative_to(ROOT)}"
            )


if __name__ == "__main__":
    print("=== 경고(2) 템플릿 기반 데이터 합성 ===")
    synthesize()
