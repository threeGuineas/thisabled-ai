#!/usr/bin/env python3
"""Build deterministic binary SAFE hard-case augmentation without external APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dedup import find_duplicate_indices  # noqa: E402

NORMAL_BASES = {
    "warm_normal": [
        "회의 끝나고 피곤해 보이던데 저녁은 천천히 챙겨 먹어.",
        "감기는 좀 나았어? 무리하지 말고 쉬어.",
        "요즘 일정이 많아 보여서 걱정돼. 답장은 여유 있을 때 해.",
        "검사 기다리느라 긴장했겠다. 이야기하고 싶으면 들어 줄게.",
        "오늘 표정이 안 좋아 보였어. 혼자 있고 싶으면 편히 쉬어.",
        "식사는 거르지 않았지? 필요한 게 있으면 알려 줘.",
        "긴 하루였을 텐데 집에서 편안히 쉬었으면 좋겠다.",
        "새로운 환경에 적응하느라 힘들지? 네 속도대로 해도 돼.",
    ],
    "accessibility_help": [
        "휠체어 출입구 위치를 확인해서 모임 단체방에 공유할게.",
        "자막 제공 회차로 예매했는데 다른 시간이 좋으면 바꿀게.",
        "교실을 옮길 때 도움이 필요하면 선생님과 같이 도와줄게.",
        "버스가 붐비면 다음 차를 타도 되도록 시간을 넉넉히 잡자.",
        "자료 글씨가 작으면 확대본을 참석자 모두에게 보내겠습니다.",
        "엘리베이터 점검 시간을 확인해서 다른 이동 경로도 알아볼게.",
        "수어 통역 신청 방법을 행사 담당자에게 문의해 두었어.",
        "조용히 쉴 공간이 필요하면 안내 데스크에 함께 물어보자.",
    ],
    "school_work": [
        "내가 오전 업무를 맡을 테니 예정된 진료를 다녀와.",
        "회의록은 정리한 다음 참석자 전원에게 공유하겠습니다.",
        "수행평가 자료는 반 친구들이 볼 수 있게 단체방에 올렸어.",
        "보고서 초안을 읽어 봤는데 표 제목만 다듬으면 좋겠어.",
        "오늘 발표 설명이 명확했어. 준비를 많이 했구나.",
        "동아리 신청 마감이 금요일이라 잊지 않도록 알려 주는 거야.",
    ],
    "benign_affection": [
        "가장 친한 친구라서 든든해. 내일도 같이 점심 먹자.",
        "결혼기념일 축하해. 가족들과 행복한 저녁 보내.",
        "새로운 일을 시작한 걸 축하하고 네 선택을 응원할게.",
        "여행 사진에서 표정이 밝아 보여서 나도 기분이 좋다.",
        "졸업을 진심으로 축하해. 반 친구들이 카드도 준비했어.",
        "항상 내 이야기를 들어 줘서 고마워. 나도 네 편이 될게.",
    ],
    "security_boundary": [
        "문자로 온 인증번호는 누구에게도 알려주지 말고 직접 확인해.",
        "환급을 빌미로 수수료를 요구하면 송금하지 말고 신고하세요.",
    ],
}

RISK_BASES = {
    "fraud_credentials": [
        "본인 확인을 위해 문자 인증 코드를 불러 주세요.",
        "환급을 받으려면 카드 번호와 비밀번호를 입력하세요.",
        "당첨금 지급 전에 세금 명목의 돈을 먼저 송금하세요.",
        "계정 정지를 막으려면 링크에서 금융 정보를 갱신하세요.",
        "지원금 신청을 대신할 테니 공동인증서 암호를 보내 주세요.",
        "대출 실행을 위해 원격 제어 앱을 휴대폰에 설치하세요.",
        "택배 주소 확인용으로 결제 승인 번호를 알려 주세요.",
        "고수익을 보장하니 지정 계좌로 투자금을 이체하세요.",
        "환불 처리에 필요하니 계좌번호와 보안카드를 촬영해 보내세요.",
        "수사기관 안전 계좌로 예금을 전부 옮겨야 합니다.",
    ],
    "grooming": [
        "부모님이 모르게 우리끼리 따로 만나자.",
        "선생님에게는 말하지 말고 혼자 약속 장소로 와.",
        "보호자가 잠든 뒤 카메라를 켜서 사진을 보내 줘.",
        "우리 대화는 특별하니까 기록을 모두 삭제해.",
        "친구들과 연락을 끊으면 원하는 선물을 사 줄게.",
        "가족보다 내가 너를 잘 아니까 나한테만 의지해.",
        "집 주소와 학교 이름을 알려 주면 몰래 데리러 갈게.",
        "사진을 보내지 않으면 지금까지의 대화를 공개할 거야.",
    ],
    "coercive_control": [
        "누구와 있는지 확인하게 실시간 위치를 계속 공유해.",
        "내 허락 없이 친구를 만나면 휴대폰을 빼앗을 거야.",
        "휴대폰 잠금번호를 말하고 통화 목록도 매일 보여 줘.",
        "내가 정한 사람 외에는 연락하지 말고 바로 답장해.",
        "사진 보내기 싫다고 해도 거절하지 말고 시키는 대로 해.",
        "헤어지면 직장에 찾아가서 곤란하게 만들겠다.",
    ],
    "threat": [
        "내 요구를 계속 무시하면 퇴근길을 막고 해치겠다.",
        "신고하면 네 가족이 사는 곳으로 찾아갈 거야.",
        "문을 열지 않으면 부수고 들어가 버리겠다.",
        "단체방을 나가면 네 비밀을 모두 퍼뜨릴 거야.",
        "학교 앞에서 기다렸다가 가만두지 않겠다.",
        "헤어지자는 말을 하면 누구도 무사하지 못해.",
    ],
}

NORMAL_V2_BASES = {
    "financial_benign": [
        "미납 안내 문자는 공식 고객센터에 직접 전화해서 확인하세요.",
        "정부 지원금은 처리 비용을 먼저 요구하지 않는다고 합니다.",
        "은행 상담 중에도 화면 공유 앱은 실행하지 마세요.",
        "환불은 판매처 공식 앱의 주문 내역에서 신청했습니다.",
        "부모님 카드로 결제하기 전에 반드시 허락을 받아야 해.",
    ],
    "benign_guidance": [
        "귀가 시간은 가족과 상의해서 안전한 시간으로 정하자.",
        "교복 선택은 학교 규정을 확인한 뒤 네가 편한 것으로 골라.",
        "등하교 경로는 반 친구들과 함께 안전 지도를 만들자.",
        "숙제하기 어려운 부분은 선생님께 도움을 요청해도 돼.",
        "촬영된 영상 삭제는 서비스 고객센터에 정식으로 요청하세요.",
    ],
}

RISK_V2_BASES = {
    "fraud_credentials": [
        "연체 취소를 위해 문자에 나온 승인 숫자를 상담원에게 말하세요.",
        "보상금 지급 대상이니 행정 처리비를 먼저 납부하십시오.",
        "배송비 환불을 도와줄 테니 화면 공유를 켜고 은행 앱을 여세요.",
        "요금 오류를 고치려면 휴대폰 원격 지원 프로그램을 설치하세요.",
        "세금 환급 계좌 등록에 필요하니 카드 뒷면 숫자를 보내 주세요.",
    ],
    "routine_recon": [
        "혼자 등교하는 시간과 매일 지나가는 길을 자세히 알려 줘.",
        "보호자가 집을 비우는 요일과 돌아오는 시간을 적어 보내.",
        "방과 후 누구와 어디에 있는지 매일 나에게 보고해.",
        "친구 없이 혼자 있는 장소를 지도에 표시해서 보내 줘.",
        "학교에서 집까지 이동하는 모습을 영상으로 계속 보여 줘.",
    ],
    "coercive_control": [
        "입을 옷과 외출 시간을 내가 정할 테니 토 달지 마.",
        "머리 모양도 내 허락을 받고 바꾸고 약속은 전부 취소해.",
        "내가 고른 옷이 아니면 밖에 나가지 못하게 할 거야.",
        "친구를 만나는 날짜와 귀가 시간을 내가 승인해야 해.",
        "네 일정표를 매일 보내고 허락받지 않은 곳에는 가지 마.",
    ],
    "digital_extortion": [
        "녹화 영상을 삭제해 주는 대가로 매달 돈을 보내.",
        "사진을 퍼뜨리지 않는 조건으로 내 과제를 대신 끝내.",
        "채팅 내용을 공개하기 싫으면 매주 내가 시키는 일을 해.",
        "카메라 영상 유포를 막으려면 정기적으로 상품권을 사 보내.",
        "계정을 돌려받고 싶으면 내 일을 무료로 계속 도와줘.",
    ],
}

# v3: blind v4 회귀에서 드러난 약점(fraud_credentials·coercive_control recall 0.75,
# 정당한 삭제·환급 안내를 위험으로 본 FP)을 정조준한다. blind v1~v4 문장과는
# 어휘를 달리해 near-dup(0.8) 누수 가드를 통과하도록 작성했다.
NORMAL_V3_BASES = {
    "records_benign": [
        "촬영본 삭제는 기관 민원실에 정식 서식으로 요청해 두었어요.",
        "열람하신 자료의 파기는 담당 부서에 공문으로 접수됩니다.",
        "분실 신고 접수증 번호로 처리 상황을 조회하실 수 있어요.",
        "회의 기록 수정은 총무팀 공식 메일로 요청하면 반영됩니다.",
        "개인정보 열람 청구서는 홈페이지 서식을 내려받아 제출하세요.",
    ],
    "financial_benign_v3": [
        "환급은 선입금을 요구하지 않으니 관공서에 직접 확인하세요.",
        "인증 문자 숫자는 어떤 상담원에게도 불러 주면 안 됩니다.",
        "요금 정정은 통신사 공식 번호로 전화해 처리하시면 됩니다.",
        "지원 대상 안내가 와도 개인정보는 공식 창구에서만 확인하세요.",
    ],
}

RISK_V3_BASES = {
    "fraud_credentials": [
        "환급 처리해 줄 테니 방금 받은 인증 숫자를 불러 주세요.",
        "요금 정정하게 휴대폰에 뜬 확인 코드를 읽어 주면 돼요.",
        "재배송하려면 문자에 온 숫자 여섯 개를 알려 주시면 됩니다.",
        "계정 잠금을 풀려면 전송된 보안 코드를 그대로 보내 주세요.",
        "당첨 확인용으로 방금 도착한 인증 번호를 말해 주십시오.",
    ],
    "coercive_control": [
        "밖에서 입을 옷은 내가 골라 줄 테니 나머지는 처분해.",
        "머리 모양이나 화장은 나한테 먼저 허락받고 바꿔.",
        "내가 사 준 것만 입고 다른 옷은 다 내다 버려.",
        "오늘 어디서 누굴 만날지 시간표를 나한테 승인받아.",
        "약속 잡기 전에 상대가 누군지 전부 나에게 보고해.",
    ],
}

PREFIXES = ["", "오늘 ", "중요한 얘기인데 ", "다시 말하지만 "]
NORMAL_SUFFIXES = ["", " 편할 때 확인해.", "부담되면 말해 줘.", "답장은 나중에 해도 괜찮아."]
RISK_SUFFIXES = ["", "지금 바로 해.", "거절하면 후회할 거야.", "다른 사람에게 알리면 안 돼."]


def normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", text.lower())


def build_rows(include_v2: bool = False, include_v3: bool = False) -> list[dict]:
    # v3는 v2를, v2는 v1을 누적한다 (라운드가 올라갈수록 이전 hard-case 전부 포함).
    if include_v3:
        include_v2 = True
    version = "v3" if include_v3 else "v2" if include_v2 else "v1"
    rows: list[dict] = []
    groups_to_build = [
        (0, NORMAL_BASES, NORMAL_SUFFIXES),
        (1, RISK_BASES, RISK_SUFFIXES),
    ]
    if include_v2:
        groups_to_build += [
            (0, NORMAL_V2_BASES, NORMAL_SUFFIXES),
            (1, RISK_V2_BASES, RISK_SUFFIXES),
        ]
    if include_v3:
        groups_to_build += [
            (0, NORMAL_V3_BASES, NORMAL_SUFFIXES),
            (1, RISK_V3_BASES, RISK_SUFFIXES),
        ]
    for label, groups, suffixes in groups_to_build:
        for slice_name, bases in groups.items():
            for base in bases:
                for prefix in PREFIXES:
                    for suffix in suffixes:
                        text = f"{prefix}{base}{suffix}".strip()
                        rows.append(
                            {
                                "text": text,
                                "label": label,
                                "slice": slice_name,
                                "source": f"safe_hardcase_{version}",
                            }
                        )
    unique: dict[str, dict] = {}
    for row in rows:
        unique.setdefault(normalize(row["text"]), row)
    result = list(unique.values())
    for index, row in enumerate(result):
        row["source_id"] = f"safe_hardcase_{version}_{index:05d}"
    return result


def read_jsonl_texts(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        json.loads(line)["text"] for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/synthetic/safe_hardcases/train.jsonl",
    )
    parser.add_argument("--include-v2", action="store_true")
    parser.add_argument("--include-v3", action="store_true")
    parser.add_argument(
        "--forbidden",
        type=Path,
        action="append",
        default=[ROOT / "tests/fixtures/safe_blind_v1.jsonl"],
    )
    args = parser.parse_args()

    rows = build_rows(include_v2=args.include_v2, include_v3=args.include_v3)
    forbidden = [text for path in args.forbidden for text in read_jsonl_texts(path)]
    forbidden_exact = {normalize(text) for text in forbidden}
    exact = [row["source_id"] for row in rows if normalize(row["text"]) in forbidden_exact]
    near = find_duplicate_indices(forbidden, [row["text"] for row in rows], threshold=0.8)
    if exact or near:
        raise RuntimeError(f"forbidden overlap: exact={exact}, near={sorted(near)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    args.output.write_text(payload, encoding="utf-8")
    counts = {label: sum(row["label"] == label for row in rows) for label in (0, 1)}
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(rows),
                "labels": counts,
                "forbidden_exact": len(exact),
                "forbidden_near_0.8": len(near),
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
