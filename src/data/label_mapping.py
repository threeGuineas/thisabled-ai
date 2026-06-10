"""시드 데이터셋(UnSmile, KOLD) → 4단계 위험도 라벨 매핑.

매핑 정책의 정의·근거·알려진 제약은 [docs/label_mapping.md](../../docs/label_mapping.md) 참조.
요약:
    - 0/1/2: 한 문장의 혐오 강도 축
    - 3 (긴급): 그루밍·유인·협박 등 행위 기반 위험 — 시드에 0건, 합성으로 보강
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

LABEL_NORMAL = 0
LABEL_CAUTION = 1
LABEL_WARNING = 2
LABEL_EMERGENCY = 3

LABEL_NAMES = ["정상", "주의", "경고", "긴급"]

UNSMILE_HATE_COLS: list[str] = [
    "여성/가족",
    "남성",
    "성소수자",
    "인종/국적",
    "연령",
    "지역",
    "종교",
    "기타 혐오",
]
UNSMILE_PROFANITY_COL = "악플/욕설"
UNSMILE_CLEAN_COL = "clean"
UNSMILE_TEXT_COL = "문장"


def unsmile_row_to_label(row: Mapping[str, Any]) -> int:
    """UnSmile 한 행 → 4단계 라벨.

    우선순위: clean → 그룹 혐오 → 악플/욕설 → 0 (fallback).
    """
    if row[UNSMILE_CLEAN_COL] == 1:
        return LABEL_NORMAL
    if any(row[c] == 1 for c in UNSMILE_HATE_COLS):
        return LABEL_WARNING
    if row[UNSMILE_PROFANITY_COL] == 1:
        return LABEL_CAUTION
    return LABEL_NORMAL


def kold_record_to_label(rec: Mapping[str, Any]) -> int:
    """KOLD 한 항목 → 4단계 라벨.

    규칙: OFF=False → 0, OFF=True & TGT=group → 2, 그 외 OFF=True → 1.
    """
    if not rec["OFF"]:
        return LABEL_NORMAL
    if rec["TGT"] == "group":
        return LABEL_WARNING
    return LABEL_CAUTION


def aihub_record_to_label(rec: Mapping[str, Any]) -> int:
    """AI-Hub 558(텍스트 윤리검증 데이터) 한 항목 → 4단계 라벨.

    규칙 (Option B - 긴급 재정의):
    1) is_immoral == False → 0 (정상)
    2) types에 SEXUAL, CRIME, VIOLENCE 포함 & intensity >= 2.0 → 3 (긴급)
    3) types에 HATE, DISCRIMINATION, CENSURE, ABUSE 포함 & intensity >= 2.0 → 2 (경고)
    4) 위 조건에 안 걸린 비윤리 문장 전부 (catch-all else) → 1 (주의)
    """
    if not rec.get("is_immoral", False):
        return LABEL_NORMAL

    types = set(rec.get("types", []))
    intensity = float(rec.get("intensity", 0.0))

    if intensity >= 2.0:
        if bool(types.intersection({"SEXUAL", "CRIME", "VIOLENCE"})):
            return LABEL_EMERGENCY
        if bool(types.intersection({"HATE", "DISCRIMINATION", "CENSURE", "ABUSE"})):
            return LABEL_WARNING

    # Catch-all fallback for any other immoral sentences (including intensity 1.x)
    return LABEL_CAUTION


def unsmile_dataframe_to_labels(df: pd.DataFrame) -> pd.Series:
    """UnSmile DataFrame 벡터화 매핑."""
    labels = pd.Series(LABEL_NORMAL, index=df.index, dtype=np.int8)
    hate_count = df[UNSMILE_HATE_COLS].sum(axis=1)
    is_profanity_only = (df[UNSMILE_PROFANITY_COL] == 1) & (hate_count == 0)
    labels.loc[is_profanity_only] = LABEL_CAUTION
    labels.loc[hate_count >= 1] = LABEL_WARNING
    labels.loc[df[UNSMILE_CLEAN_COL] == 1] = LABEL_NORMAL
    return labels


def kold_dataframe_to_labels(df: pd.DataFrame) -> pd.Series:
    """KOLD DataFrame 벡터화 매핑."""
    labels = pd.Series(LABEL_NORMAL, index=df.index, dtype=np.int8)
    is_off = df["OFF"].astype(bool)
    is_group = df["TGT"] == "group"
    labels.loc[is_off & ~is_group] = LABEL_CAUTION
    labels.loc[is_off & is_group] = LABEL_WARNING
    return labels


def build_unified_dataset(
    unsmile_df: pd.DataFrame,
    kold_df: pd.DataFrame,
) -> pd.DataFrame:
    """두 시드 데이터셋을 통합 스키마로 변환.

    Returns:
        ``[text, label, source, source_id]`` 컬럼의 DataFrame.
    """
    us = pd.DataFrame(
        {
            "text": unsmile_df[UNSMILE_TEXT_COL].astype(str),
            "label": unsmile_dataframe_to_labels(unsmile_df).astype(np.int8),
            "source": "unsmile",
            "source_id": unsmile_df.index.astype(str),
        }
    )
    ko = pd.DataFrame(
        {
            "text": kold_df["comment"].astype(str),
            "label": kold_dataframe_to_labels(kold_df).astype(np.int8),
            "source": "kold",
            "source_id": kold_df["guid"].astype(str)
            if "guid" in kold_df
            else kold_df.index.astype(str),
        }
    )
    return pd.concat([us, ko], ignore_index=True)
