"""P1.5 — AI허브 실문장으로 MATCH v2 관심 태그별 문장 풀을 만든다.

소스 (data/raw/aihub_558, 원문 재배포 금지 → 가공 산출물만 저장):
- 020.주제별 텍스트 일상 대화 데이터: 취미 태그별 키워드 매칭으로 실제 발화문 수집.
  annotations.lines[].norm_text(화자 prefix 없는 정규화문)를 사용한다.
- 147.텍스트 윤리검증 데이터: is_immoral=false 문장만 골라 일반 대화 태그
  (daily_sharing/small_talk/heart_sharing/healing) 풀로 사용. 독성 문장은 제외한다.

출력: data/processed/match_corpus.json = {tag_id: [문장, ...]}  (태그당 상한/중복제거).
이 파일이 build_profiles_v2의 실문장 소스가 된다(없으면 자체 템플릿으로 폴백).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_profiles_v2 import load_allowed_tags  # noqa: E402
from src.data.matching_input import contains_contact_info  # noqa: E402

RAW_558 = ROOT / "data" / "raw" / "aihub_558"
DIR_020 = RAW_558 / "020.주제별 텍스트 일상 대화 데이터" / "01.데이터"
DIR_147 = RAW_558 / "147.텍스트 윤리검증 데이터" / "01.데이터"
OUT_PATH = ROOT / "data" / "processed" / "match_corpus.json"

MIN_LEN = 6
MAX_LEN = 100
GENERAL_TAGS = ("daily_sharing", "small_talk", "heart_sharing", "healing")

# 취미 태그별 키워드. norm_text에 부분 문자열로 포함되면 그 태그 풀에 넣는다.
TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "walking": ("산책", "걷기", "걷는", "걸으"),
    "hiking": ("등산", "산행", "트레킹", "등반"),
    "gym": ("헬스", "웨이트", "근력", "헬스장", "피티"),
    "yoga": ("요가", "필라테스", "스트레칭"),
    "cycling": ("자전거", "라이딩", "따릉이", "사이클"),
    "home_training": ("홈트", "홈트레이닝", "맨몸 운동", "맨몸운동"),
    "mobile_game": ("모바일 게임", "모바일게임", "폰 게임", "핸드폰 게임", "앱 게임"),
    "pc_game": (
        "피시방",
        "피씨방",
        "pc방",
        "컴퓨터 게임",
        "온라인 게임",
        "롤 ",
        "오버워치",
        "배그",
        "스팀",
    ),
    "board_game": ("보드게임", "보드 게임", "루미큐브", "할리갈리", "부루마블", "카탄"),
    "puzzle": ("퍼즐", "직소", "큐브 맞추"),
    "listening_music": ("음악 듣", "노래 듣", "플레이리스트", "음악 감상", "음악감상"),
    "singing": ("노래방", "노래 부르", "코인노래방", "보컬"),
    "instrument": ("기타 치", "피아노", "드럼", "바이올린", "악기", "우쿨렐레"),
    "kpop": ("아이돌", "케이팝", "k-pop", "콘서트", "최애", "덕질", "방탄", "bts"),
    "trot": ("트로트", "미스트롯", "임영웅"),
    "movie": ("영화", "극장", "개봉", "영화관"),
    "drama": ("드라마", "정주행", "회차"),
    "ott": ("넷플릭스", "넷플", "디즈니플러스", "왓챠", "티빙", "ott", "스트리밍"),
    "animation": ("애니메이션", "애니 ", "지브리", "극장판"),
    "reading": ("독서", "책 읽", "소설 읽", "도서관", "북카페", "책방"),
    "webtoon": ("웹툰",),
    "webnovel": ("웹소설", "판무", "로판"),
    "writing": ("글쓰기", "일기 쓰", "블로그 글", "에세이"),
    "drawing": ("그림 그리", "드로잉", "스케치", "그림을"),
    "photo": ("사진 찍", "출사", "카메라", "인생샷", "사진을"),
    "craft_diy": ("만들기", "diy", "공예", "손재주"),
    "knitting": ("뜨개", "코바늘", "손뜨개"),
    "cooking": ("요리", "집밥", "레시피", "요리하"),
    "baking": ("베이킹", "빵 굽", "쿠키 만들", "오븐"),
    "food_tour": ("맛집", "먹방", "맛있는 집"),
    "cafe": ("카페", "아메리카노", "커피 마시", "카페 투어"),
    "dessert": ("디저트", "케이크", "마카롱", "빙수", "달달한"),
    "domestic_travel": ("여행", "당일치기", "여행지", "여행 갔"),
    "camping": ("캠핑", "차박", "글램핑", "텐트"),
    "exhibition_museum": ("전시", "미술관", "박물관", "전시회"),
    "driving": ("드라이브", "운전", "차 끌고"),
    "dog": ("강아지", "반려견", "멍멍", "산책시켜"),
    "cat": ("고양이", "냥이", "반려묘", "집사"),
    "plant": ("식물", "화분", "다육", "가드닝", "반려식물"),
    "daily_sharing": ("일상", "오늘 하루", "하루 종일"),
    "small_talk": ("수다", "잡담", "이야기하"),
    "heart_sharing": ("고민", "속상", "위로", "응원", "힘들 때"),
    "healing": ("힐링", "쉬고 싶", "마음이", "여유"),
}


# AI허브 익명화 마스크(이름 자리): "**", "***", 뒤에 조사가 붙은 "**가/이/는/의/님" 등.
_MASK_PATTERN = re.compile(r"\*+\s*[가이은는을를의랑과와도님아야씨]?")


def _clean(text: str) -> str | None:
    text = _MASK_PATTERN.sub(" ", text)
    text = " ".join(text.split())
    if "*" in text:  # 정리 후에도 마스크 잔여가 있으면 버린다.
        return None
    if not (MIN_LEN <= len(text) <= MAX_LEN):
        return None
    if contains_contact_info(text):
        return None
    return text


def _match_tags(text: str) -> list[str]:
    low = text.lower()
    return [tag for tag, kws in TAG_KEYWORDS.items() if any(kw in low for kw in kws)]


def _iter_020_norm_texts(cap_files: int | None):
    label_dir = DIR_020 / "1.Training" / "라벨링데이터"
    zips = sorted(p for p in label_dir.glob("*.zip"))
    seen = 0
    for zpath in zips:
        with zipfile.ZipFile(zpath) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".json"):
                    continue
                try:
                    doc = json.loads(zf.read(name))
                except (json.JSONDecodeError, KeyError):
                    continue
                for info in doc.get("info", []):
                    ann = info.get("annotations", {})
                    for line in ann.get("lines", []):
                        yield line.get("norm_text") or line.get("text") or ""
                seen += 1
                if cap_files and seen >= cap_files:
                    return


def _iter_147_normal_texts(cap_files: int | None):
    ext = DIR_147 / "1.Training" / "라벨링데이터" / "aihub" / "extracted"
    files = sorted(ext.rglob("*.json"))
    seen = 0
    for fpath in files:
        try:
            docs = json.loads(fpath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for doc in docs:
            for sent in doc.get("sentences", []):
                if sent.get("is_immoral") is True:
                    continue
                yield sent.get("text") or sent.get("origin_text") or ""
        seen += 1
        if cap_files and seen >= cap_files:
            return


def build_corpus(
    per_tag_cap: int, cap_020: int | None, cap_147: int | None
) -> dict[str, list[str]]:
    tags = load_allowed_tags()
    pools: dict[str, set[str]] = {tag: set() for tag in tags}

    # 020: 취미 키워드 매칭
    for raw in _iter_020_norm_texts(cap_020):
        text = _clean(raw)
        if text is None:
            continue
        for tag in _match_tags(text):
            if tag in pools and len(pools[tag]) < per_tag_cap:
                pools[tag].add(text)

    # 147: is_immoral=false → 일반 대화 태그 보충(취미 키워드에 걸리지 않는 문장만)
    for raw in _iter_147_normal_texts(cap_147):
        text = _clean(raw)
        if text is None or _match_tags(text):
            continue
        for tag in GENERAL_TAGS:
            if len(pools[tag]) < per_tag_cap:
                pools[tag].add(text)

    return {tag: sorted(sents) for tag, sents in pools.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-tag-cap", type=int, default=800)
    parser.add_argument(
        "--cap-020", type=int, default=None, help="처리할 020 대화 파일 상한(디버그)"
    )
    parser.add_argument("--cap-147", type=int, default=None, help="처리할 147 파일 상한(디버그)")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    if not DIR_020.exists():
        raise SystemExit(f"020 데이터 경로 없음: {DIR_020}")

    corpus = build_corpus(args.per_tag_cap, args.cap_020, args.cap_147)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(corpus, ensure_ascii=False, indent=1), encoding="utf-8")

    counts = {tag: len(sents) for tag, sents in corpus.items()}
    total = sum(counts.values())
    sparse = {tag: n for tag, n in counts.items() if n < 20}
    print(f"corpus written: {args.out}")
    print(f"tags={len(counts)} total_sentences={total}")
    print("per-tag counts:")
    for tag in sorted(counts, key=lambda t: counts[t]):
        print(f"  {counts[tag]:5d}  {tag}")
    if sparse:
        print(f"\nSPARSE (<20, 템플릿 폴백 유지): {sorted(sparse)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
