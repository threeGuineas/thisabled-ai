"""모듈 ① 공정성(Fairness) 평가 — 보호집단별 F1 격차.

UnSmile 7집단 + KOLD GRP top-7 + 장애 도메인.

[근거]
- label_mapping.md §6.L4 (경고(2) 정의 이질성 — source별 격차)
- baseline.md §4 (UnSmile vs KOLD 격차 측정)
- final.md 이전 보고 (집단별 격차 0.14~0.22 → 정직한 측정·개선 필요)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

UNSMILE_GROUPS = [
    "여성/가족",
    "남성",
    "성소수자",
    "인종/국적",
    "연령",
    "지역",
    "종교",
]

DISABILITY_KEYWORDS = [
    "장애",
    "장애인",
    "휠체어",
    "활동지원",
    "발달장애",
    "시각장애",
    "청각장애",
    "지체장애",
    "정신장애",
    "자폐",
]

MIN_SAMPLES = 30  # 통계적 신뢰를 위한 최소 샘플 수


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 (보유 라벨에 한정)."""
    if len(y_true) == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def evaluate_unsmile_groups(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    unsmile_raw: pd.DataFrame,
) -> dict[str, Any]:
    """UnSmile 7개 보호집단별 F1.

    test_df의 source=='unsmile' row를 source_id로 raw에 join하여 어떤 집단인지 찾음.
    """
    mask = (test_df["source"] == "unsmile").values
    if not mask.any():
        return {"groups": {}, "max_gap": 0.0, "note": "no unsmile in test"}

    us_test = test_df[mask].reset_index(drop=True).copy()
    us_pred = y_pred[mask]
    us_test["pred"] = us_pred
    # source_id는 0-based row index in raw concat(train, valid)
    us_test["raw_idx"] = us_test["source_id"].astype(int)

    # raw에 7집단 컬럼 join
    raw_subset = unsmile_raw[UNSMILE_GROUPS].copy()
    raw_subset = raw_subset.reset_index(drop=True)
    joined = us_test.join(raw_subset, on="raw_idx", how="left")

    results: dict[str, Any] = {}
    f1_values = []
    for group in UNSMILE_GROUPS:
        in_group = (joined[group] == 1).values
        n = int(in_group.sum())
        if n < MIN_SAMPLES:
            results[group] = {"n": n, "f1": None, "skipped": "n<30"}
            continue
        labels = joined["label"].values[in_group]
        preds = joined["pred"].values[in_group]
        f1 = _f1_macro(labels, preds)
        results[group] = {"n": n, "f1": f1}
        f1_values.append(f1)

    return {
        "groups": results,
        "max_gap": float(max(f1_values) - min(f1_values)) if len(f1_values) >= 2 else 0.0,
        "n_groups_measured": len(f1_values),
    }


def evaluate_kold_groups(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    kold_raw: pd.DataFrame,
    top_k: int = 7,
) -> dict[str, Any]:
    """KOLD GRP top-K 집단별 F1."""
    mask = (test_df["source"] == "kold").values
    if not mask.any():
        return {"groups": {}, "max_gap": 0.0, "note": "no kold in test"}

    ko_test = test_df[mask].reset_index(drop=True).copy()
    ko_pred = y_pred[mask]
    ko_test["pred"] = ko_pred

    # KOLD guid로 join
    kold_lookup = kold_raw.set_index("guid")["GRP"].to_dict()
    ko_test["grp"] = ko_test["source_id"].map(kold_lookup)

    # GRP top-K
    grp_counts = ko_test["grp"].value_counts(dropna=True)
    top_groups = grp_counts.head(top_k).index.tolist()

    results: dict[str, Any] = {}
    f1_values = []
    for group in top_groups:
        in_group = (ko_test["grp"] == group).values
        n = int(in_group.sum())
        if n < MIN_SAMPLES:
            results[group] = {"n": n, "f1": None, "skipped": "n<30"}
            continue
        labels = ko_test["label"].values[in_group]
        preds = ko_test["pred"].values[in_group]
        f1 = _f1_macro(labels, preds)
        results[group] = {"n": n, "f1": f1}
        f1_values.append(f1)

    return {
        "groups": results,
        "max_gap": float(max(f1_values) - min(f1_values)) if len(f1_values) >= 2 else 0.0,
        "n_groups_measured": len(f1_values),
    }


def evaluate_disability_domain(test_df: pd.DataFrame, y_pred: np.ndarray) -> dict[str, Any]:
    """장애 키워드 포함 여부별 F1."""
    df = test_df.copy()
    df["pred"] = y_pred
    df["has_disability"] = (
        df["text"].astype(str).apply(lambda t: any(kw in t for kw in DISABILITY_KEYWORDS))
    )

    results: dict[str, Any] = {}
    f1_values = []
    for label, mask in [
        ("disability_yes", df["has_disability"]),
        ("disability_no", ~df["has_disability"]),
    ]:
        n = int(mask.sum())
        if n < MIN_SAMPLES:
            results[label] = {"n": n, "f1": None, "skipped": "n<30"}
            continue
        labels = df["label"].values[mask.values]
        preds = df["pred"].values[mask.values]
        f1 = _f1_macro(labels, preds)
        results[label] = {"n": n, "f1": f1}
        f1_values.append(f1)

    return {
        "groups": results,
        "max_gap": float(max(f1_values) - min(f1_values)) if len(f1_values) >= 2 else 0.0,
    }


def run_full_fairness_evaluation(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    unsmile_raw_train_path: Path,
    unsmile_raw_valid_path: Path,
    kold_raw_path: Path,
) -> dict[str, Any]:
    """모든 grouping 통합 평가 → 단일 dict 반환."""
    unsmile_raw = pd.concat(
        [
            pd.read_csv(unsmile_raw_train_path, sep="\t"),
            pd.read_csv(unsmile_raw_valid_path, sep="\t"),
        ],
        ignore_index=True,
    )
    with kold_raw_path.open(encoding="utf-8") as f:
        kold_raw = pd.DataFrame(json.load(f))

    return {
        "unsmile_7_groups": evaluate_unsmile_groups(test_df, y_pred, unsmile_raw),
        "kold_top_groups": evaluate_kold_groups(test_df, y_pred, kold_raw),
        "disability_domain": evaluate_disability_domain(test_df, y_pred),
    }
