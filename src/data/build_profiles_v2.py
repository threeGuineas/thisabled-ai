"""MATCH 모듈② v2 재학습용 합성 프로필·콘텐츠 생성기 (P1).

설계 원칙 (outputs/10_thisabled-ai/보고서/20260721_thisabled_MATCH랭커재학습계획.md):
- 사용자마다 **잠재 관심 분포**(latent)를 먼저 뽑고, bio·태그·콘텐츠는 그 잠재값의
  노이즈 낀 실현으로 생성한다. 관측 스냅샷(UserSnapshot)에는 잠재값을 넣지 않는다.
- 라벨(P2)은 관측 특성이 아니라 잠재값에서만 계산한다(EXP-2 라벨 순환성 방지).
- 생성 스냅샷은 전수가 matching_input.validate_user_snapshot을 통과해야 하며 금지
  필드(지역·장애유형·생년월일·닉네임 등)를 만들지 않는다.

이 모듈은 SBERT/LightGBM에 의존하지 않는다. 임베딩·특성·학습은 P3 노트북이 담당한다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

from src.data.matching_input import (
    ContentSignal,
    MatchingInputPolicy,
    UserSnapshot,
    age_band_for_age,
    validate_user_snapshot,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "module2_matching.yaml"
# P1.5 scripts/adapt_match_corpus.py 산출물(실문장 풀). 없으면 템플릿으로 폴백.
DEFAULT_CORPUS_PATH = ROOT / "data" / "processed" / "match_corpus.json"

# 각 TAG-01 코드의 한국어 표현 어구. 잠재 관심 → bio/콘텐츠 문장 실현에 쓴다.
# 값은 재배포 제약이 없는 자체 작성 어구다(P1.5 AI허브 코퍼스 수령 시 확장 대체).
TAG_PHRASES: dict[str, tuple[str, ...]] = {
    "walking": ("산책", "걷기", "동네 한 바퀴"),
    "hiking": ("등산", "산행", "트레킹"),
    "gym": ("헬스", "웨이트", "운동"),
    "yoga": ("요가", "스트레칭", "필라테스"),
    "cycling": ("자전거", "라이딩", "따릉이"),
    "home_training": ("홈트", "집에서 운동", "맨몸 운동"),
    "mobile_game": ("모바일 게임", "폰 게임", "캐주얼 게임"),
    "pc_game": ("PC 게임", "컴퓨터 게임", "온라인 게임"),
    "board_game": ("보드게임", "테이블 게임", "카드 게임"),
    "puzzle": ("퍼즐", "직소 퍼즐", "두뇌 퍼즐"),
    "listening_music": ("음악 감상", "노래 듣기", "플레이리스트"),
    "singing": ("노래 부르기", "노래방", "보컬 연습"),
    "instrument": ("악기 연주", "기타 치기", "피아노"),
    "kpop": ("케이팝", "아이돌", "K-POP"),
    "trot": ("트로트", "성인가요", "트로트 무대"),
    "movie": ("영화", "영화 감상", "극장 나들이"),
    "drama": ("드라마", "드라마 정주행", "주말 드라마"),
    "ott": ("OTT", "넷플릭스", "스트리밍"),
    "animation": ("애니메이션", "애니", "극장판"),
    "reading": ("독서", "책 읽기", "북 카페"),
    "webtoon": ("웹툰", "웹툰 보기", "연재 웹툰"),
    "webnovel": ("웹소설", "장편 소설", "연재 소설"),
    "writing": ("글쓰기", "일기 쓰기", "짧은 글"),
    "drawing": ("그림 그리기", "드로잉", "스케치"),
    "photo": ("사진 찍기", "출사", "사진"),
    "craft_diy": ("만들기", "DIY", "소품 제작"),
    "knitting": ("뜨개질", "손뜨개", "코바늘"),
    "cooking": ("요리", "집밥", "새 레시피"),
    "baking": ("베이킹", "빵 굽기", "홈 베이킹"),
    "food_tour": ("맛집 탐방", "맛집 투어", "먹킷리스트"),
    "cafe": ("카페", "카페 투어", "커피"),
    "dessert": ("디저트", "케이크", "달달한 것"),
    "domestic_travel": ("국내 여행", "당일치기", "여행"),
    "camping": ("캠핑", "차박", "글램핑"),
    "exhibition_museum": ("전시 관람", "미술관", "박물관"),
    "driving": ("드라이브", "야경 드라이브", "차 타고 나들이"),
    "dog": ("강아지", "반려견", "산책 친구"),
    "cat": ("고양이", "반려묘", "냥이"),
    "plant": ("식물 키우기", "반려식물", "가드닝"),
    "daily_sharing": ("일상 공유", "소소한 일상", "하루 기록"),
    "small_talk": ("수다", "가벼운 대화", "이야기 나누기"),
    "heart_sharing": ("고민 나눔", "속마음 이야기", "서로 응원"),
    "healing": ("힐링", "쉼", "마음 챙김"),
}

_BIO_FRAMES: tuple[str, ...] = (
    "{phrase} 좋아해요.",
    "요즘 {phrase}에 푹 빠졌어요.",
    "{phrase} 같이 할 친구 찾아요.",
    "주말엔 주로 {phrase} 해요.",
    "{phrase} 이야기 나누는 거 좋아해요.",
)
_CONTENT_FRAMES: tuple[str, ...] = (
    "오늘 {phrase} 했는데 정말 좋았어요.",
    "{phrase} 관련해서 추천 좀 받고 싶어요.",
    "{phrase} 하다 보면 시간 가는 줄 몰라요.",
    "이번 주에 {phrase} 하려고요.",
    "{phrase} 취미 가진 분 있나요?",
    "{phrase} 하고 나면 기분이 좋아져요.",
)

_UI_MODES: tuple[str, ...] = ("visual", "hearing", "developmental", "")


@dataclass(frozen=True, slots=True)
class LatentUser:
    """관측 스냅샷 밖에 보관하는 라벨 계산용 잠재 상태."""

    user_id: str
    interest_weights: np.ndarray  # allowed_tag_ids 순서의 정규화 관심 분포
    age: int
    social_cluster: int


@dataclass(frozen=True, slots=True)
class SyntheticUser:
    snapshot: UserSnapshot
    latent: LatentUser


@dataclass
class GenerationConfig:
    n_users: int = 10_000
    n_social_clusters: int = 40
    core_interest_min: int = 2
    core_interest_max: int = 5
    tag_dropout: float = 0.25  # 잠재 관심 태그를 관측에서 누락시킬 확률
    tag_noise: float = 0.10  # 무관 태그를 관측에 추가할 확률
    no_bio_rate: float = 0.15
    no_authored_rate: float = 0.30
    no_liked_rate: float = 0.35
    max_authored: int = 12
    max_liked: int = 12
    lookback_days: int = 90
    minor_rate: float = 0.08  # 14~18세 비율
    ui_mode_missing_rate: float = 0.10
    age_band_only_rate: float = 0.05  # age_years 없이 age_band만 주는 비율
    use_corpus: bool = True  # 실문장 코퍼스가 있으면 콘텐츠 텍스트에 사용
    min_corpus_sentences: int = 20  # 이 미만인 태그는 템플릿으로 폴백
    seed: int = 42
    as_of: datetime = field(default_factory=lambda: datetime(2026, 7, 21, tzinfo=UTC))


def load_allowed_tags(config_path: Path | None = None) -> tuple[str, ...]:
    """config의 allowed_tag_ids를 단일 소스로 로드한다."""

    path = config_path or DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    tags = tuple(str(tag) for tag in config["input"]["allowed_tag_ids"])
    if not tags:
        raise ValueError("allowed_tag_ids must not be empty")
    missing = [tag for tag in tags if tag not in TAG_PHRASES]
    if missing:
        raise ValueError(f"TAG_PHRASES missing entries for: {missing}")
    return tags


def load_corpus(path: Path | None = None) -> dict[str, list[str]] | None:
    """P1.5 실문장 코퍼스를 로드한다. 파일이 없으면 None(템플릿 폴백)."""

    corpus_path = path or DEFAULT_CORPUS_PATH
    if not corpus_path.exists():
        return None
    with corpus_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(tag): [str(s) for s in sents] for tag, sents in data.items()}


def _sample_latent_interest(
    rng: np.random.Generator, n_tags: int, cfg: GenerationConfig
) -> np.ndarray:
    """소수 코어 관심에 질량이 몰린 희소 분포를 뽑는다."""

    n_core = int(rng.integers(cfg.core_interest_min, cfg.core_interest_max + 1))
    core_idx = rng.choice(n_tags, size=min(n_core, n_tags), replace=False)
    weights = np.zeros(n_tags, dtype=np.float64)
    weights[core_idx] = rng.dirichlet(np.ones(len(core_idx)) * 2.0)
    # 배경 잡음(모든 태그에 미세한 관심)
    weights += rng.uniform(0.0, 0.02, size=n_tags)
    return weights / weights.sum()


def _sample_age(rng: np.random.Generator, cfg: GenerationConfig) -> int:
    if rng.random() < cfg.minor_rate:
        return int(rng.integers(14, 19))
    age = int(round(rng.normal(34, 11)))
    return max(19, min(75, age))


def _observed_tags(
    rng: np.random.Generator,
    interest_weights: np.ndarray,
    tags: tuple[str, ...],
    cfg: GenerationConfig,
    max_tags: int,
) -> tuple[str, ...]:
    """잠재 관심에서 태그를 노이즈 있게 관측으로 실현한다."""

    order = np.argsort(interest_weights)[::-1]
    chosen: list[str] = []
    for idx in order:
        if interest_weights[idx] < 1e-6:
            break
        if rng.random() >= cfg.tag_dropout:
            chosen.append(tags[idx])
        if len(chosen) >= max_tags:
            break
    if rng.random() < cfg.tag_noise and len(chosen) < max_tags:
        noise_tag = tags[int(rng.integers(len(tags)))]
        if noise_tag not in chosen:
            chosen.append(noise_tag)
    if not chosen:  # 최소 1개는 보장
        chosen.append(tags[int(order[0])])
    return tuple(dict.fromkeys(chosen))[:max_tags]


def _interest_phrase(rng: np.random.Generator, weights: np.ndarray, tags: tuple[str, ...]) -> str:
    idx = int(rng.choice(len(tags), p=weights))
    phrases = TAG_PHRASES[tags[idx]]
    return phrases[int(rng.integers(len(phrases)))]


def _interest_content_text(
    rng: np.random.Generator,
    weights: np.ndarray,
    tags: tuple[str, ...],
    corpus: dict[str, list[str]] | None,
    cfg: GenerationConfig,
) -> str:
    """잠재 관심 태그의 실문장(있으면)이나 템플릿으로 콘텐츠 텍스트를 만든다."""

    idx = int(rng.choice(len(tags), p=weights))
    tag = tags[idx]
    if corpus is not None:
        pool = corpus.get(tag, [])
        if len(pool) >= cfg.min_corpus_sentences:
            return pool[int(rng.integers(len(pool)))]
    frame = _CONTENT_FRAMES[int(rng.integers(len(_CONTENT_FRAMES)))]
    return frame.format(phrase=TAG_PHRASES[tag][int(rng.integers(len(TAG_PHRASES[tag])))])


def _make_bio(
    rng: np.random.Generator, weights: np.ndarray, tags: tuple[str, ...], cfg: GenerationConfig
) -> str:
    if rng.random() < cfg.no_bio_rate:
        return ""
    n_sentences = int(rng.integers(1, 4))
    parts: list[str] = []
    for _ in range(n_sentences):
        frame = _BIO_FRAMES[int(rng.integers(len(_BIO_FRAMES)))]
        parts.append(frame.format(phrase=_interest_phrase(rng, weights, tags)))
    bio = " ".join(parts)
    return bio[:280]


def _make_content(
    rng: np.random.Generator,
    weights: np.ndarray,
    tags: tuple[str, ...],
    *,
    prefix: str,
    n_items: int,
    cfg: GenerationConfig,
    corpus: dict[str, list[str]] | None,
) -> tuple[ContentSignal, ...]:
    items: list[ContentSignal] = []
    for i in range(n_items):
        text = _interest_content_text(rng, weights, tags, corpus, cfg)
        age_days = float(rng.uniform(0, cfg.lookback_days - 1))
        created = cfg.as_of - timedelta(days=age_days)
        source = "post" if rng.random() < 0.6 else "comment"
        items.append(
            ContentSignal(
                content_id=f"{prefix}_{i}",
                source_type=source,
                text=text,
                created_at=created,
            )
        )
    return tuple(items)


def generate_user(
    rng: np.random.Generator,
    index: int,
    tags: tuple[str, ...],
    cfg: GenerationConfig,
    policy: MatchingInputPolicy,
    corpus: dict[str, list[str]] | None = None,
) -> SyntheticUser:
    user_id = f"u2_{index:06d}"
    weights = _sample_latent_interest(rng, len(tags), cfg)
    age = _sample_age(rng, cfg)
    cluster = int(rng.integers(cfg.n_social_clusters))

    observed_tags = _observed_tags(rng, weights, tags, cfg, policy.max_tags)
    bio = _make_bio(rng, weights, tags, cfg)

    n_authored = (
        0 if rng.random() < cfg.no_authored_rate else int(rng.integers(1, cfg.max_authored + 1))
    )
    n_liked = 0 if rng.random() < cfg.no_liked_rate else int(rng.integers(1, cfg.max_liked + 1))
    authored = _make_content(
        rng, weights, tags, prefix=f"{user_id}_a", n_items=n_authored, cfg=cfg, corpus=corpus
    )
    liked = _make_content(
        rng, weights, tags, prefix=f"{user_id}_l", n_items=n_liked, cfg=cfg, corpus=corpus
    )

    ui_mode = "" if rng.random() < cfg.ui_mode_missing_rate else str(rng.choice(_UI_MODES[:3]))

    # 대부분 age_years 제공(band는 검증기가 파생). 일부는 band만 제공.
    if rng.random() < cfg.age_band_only_rate:
        snapshot = UserSnapshot(
            user_id=user_id,
            bio=bio,
            tag_ids=observed_tags,
            age_years=None,
            age_band=age_band_for_age(age),
            ui_mode=ui_mode,
            authored_items=authored,
            liked_items=liked,
        )
    else:
        snapshot = UserSnapshot(
            user_id=user_id,
            bio=bio,
            tag_ids=observed_tags,
            age_years=age,
            age_band=None,
            ui_mode=ui_mode,
            authored_items=authored,
            liked_items=liked,
        )

    validated = validate_user_snapshot(snapshot, policy)
    latent = LatentUser(
        user_id=user_id,
        interest_weights=weights.astype(np.float64),
        age=age,
        social_cluster=cluster,
    )
    return SyntheticUser(snapshot=validated, latent=latent)


def generate_population(
    cfg: GenerationConfig | None = None,
    *,
    policy: MatchingInputPolicy | None = None,
    config_path: Path | None = None,
    corpus: dict[str, list[str]] | None = None,
    corpus_path: Path | None = None,
) -> list[SyntheticUser]:
    """cfg.n_users명의 합성 사용자를 결정적으로 생성한다."""

    cfg = cfg or GenerationConfig()
    tags = load_allowed_tags(config_path)
    policy = policy or MatchingInputPolicy(allowed_tag_ids=frozenset(tags))
    if corpus is None and cfg.use_corpus:
        corpus = load_corpus(corpus_path)
    rng = np.random.default_rng(cfg.seed)
    return [generate_user(rng, i, tags, cfg, policy, corpus) for i in range(cfg.n_users)]
