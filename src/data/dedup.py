"""시드↔합성 데이터 누수 차단 — MinHash LSH 기반 근사 중복 제거.

project_facts 재현성 규약: "시드-합성 간 중복 제거: MinHash 기반 유사 문장 제거로
데이터 누수 방지." 합성 train 행이 시드(train) 행과 사실상 같은 문장이면, 그 합성 행은
train에 섞여 학습되고 동시에 평가에도 새는 통로가 된다. 여기서는 문자 n-gram shingle의
MinHash Jaccard 유사도가 임계값 이상인 합성 행을 제거한다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import pandas as pd

try:
    from datasketch import MinHash, MinHashLSH
except ImportError as exc:  # pragma: no cover - 의존성 안내용
    raise ImportError(
        "MinHash 중복 제거에는 `datasketch`가 필요합니다. `pip install datasketch`."
    ) from exc

DEFAULT_THRESHOLD = 0.8
DEFAULT_NUM_PERM = 128
DEFAULT_SHINGLE_K = 3

_WS_RE = re.compile(r"\s+")


def _normalize(text: object) -> str:
    """공백 정규화 + 양끝 trim. None/NaN은 빈 문자열."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return _WS_RE.sub(" ", str(text)).strip()


def _shingles(text: str, k: int = DEFAULT_SHINGLE_K) -> set[str]:
    """문자 단위 k-gram shingle 집합. 길이 < k면 전체 문자열 1개."""
    if len(text) < k:
        return {text} if text else set()
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def _minhash(text: str, num_perm: int, k: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in _shingles(text, k):
        m.update(sh.encode("utf-8"))
    return m


def find_duplicate_indices(
    reference_texts: Iterable[object],
    candidate_texts: Sequence[object],
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    shingle_k: int = DEFAULT_SHINGLE_K,
) -> set[int]:
    """``candidate_texts`` 중 ``reference_texts``와 근사 중복인 행의 위치 인덱스 집합.

    Jaccard 유사도(추정치) ``>= threshold``면 중복으로 판정한다.

    Returns:
        candidate 리스트에서의 0-기반 위치 인덱스 집합.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    n_ref = 0
    for i, t in enumerate(reference_texts):
        norm = _normalize(t)
        if not norm:
            continue
        lsh.insert(f"ref-{i}", _minhash(norm, num_perm, shingle_k))
        n_ref += 1

    duplicates: set[int] = set()
    if n_ref == 0:
        return duplicates

    for j, t in enumerate(candidate_texts):
        norm = _normalize(t)
        if not norm:
            continue
        if lsh.query(_minhash(norm, num_perm, shingle_k)):
            duplicates.add(j)
    return duplicates


def deduplicate_against(
    reference_texts: Iterable[object],
    candidate_df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    shingle_k: int = DEFAULT_SHINGLE_K,
) -> tuple[pd.DataFrame, int]:
    """``candidate_df``에서 ``reference_texts``와 근사 중복인 행을 제거.

    Returns:
        ``(deduped_df, removed_count)`` — 위치 인덱스 reset 완료된 DataFrame과 제거 건수.
    """
    if candidate_df.empty:
        return candidate_df.reset_index(drop=True), 0

    dup_idx = find_duplicate_indices(
        reference_texts,
        candidate_df[text_col].tolist(),
        threshold=threshold,
        num_perm=num_perm,
        shingle_k=shingle_k,
    )
    if not dup_idx:
        return candidate_df.reset_index(drop=True), 0

    keep_mask = [i not in dup_idx for i in range(len(candidate_df))]
    deduped = candidate_df.iloc[keep_mask].reset_index(drop=True)
    return deduped, len(dup_idx)
