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


def _group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """집단 하나의 지표. 단일 클래스 슬라이스에서도 해석 가능한 값을 함께 낸다.

    UnSmile·KOLD의 보호집단 슬라이스는 혐오표현 대상 주석이라 이진 붕괴 후 정답이
    전부 위험(1)이다. 이때 macro-F1은 전부 맞히면 1.0, 몇 건만 놓쳐도 약 0.5로
    떨어져 집단 간 비교가 불가능하다. 그래서 위험 재현율을 함께 보고하고,
    격차 판정은 재현율로 한다.
    """

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    return {
        "n": int(len(y_true)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "single_class": n_pos == 0 or n_neg == 0,
        "f1": _f1_macro(y_true, y_pred),
        "recall": float((y_pred[y_true == 1] == 1).mean()) if n_pos else None,
    }


def _gap(values: list[float]) -> float:
    return float(max(values) - min(values)) if len(values) >= 2 else 0.0


def _summarize(results: dict[str, Any]) -> dict[str, Any]:
    """집단별 결과에서 f1/recall 격차를 뽑는다."""

    f1_values = [r["f1"] for r in results.values() if r.get("f1") is not None]
    recall_values = [r["recall"] for r in results.values() if r.get("recall") is not None]
    any_single = any(r.get("single_class") for r in results.values() if "single_class" in r)
    summary: dict[str, Any] = {
        "groups": results,
        "max_gap": _gap(f1_values),
        "max_recall_gap": _gap(recall_values),
        "n_groups_measured": len(f1_values),
        "single_class_groups": any_single,
    }
    if any_single:
        summary["note"] = (
            "정답이 한 클래스뿐인 집단이 있어 macro-F1(max_gap)은 해석 불가. "
            "격차는 max_recall_gap 으로 판정할 것."
        )
    return summary


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
    for group in UNSMILE_GROUPS:
        in_group = (joined[group] == 1).values
        n = int(in_group.sum())
        if n < MIN_SAMPLES:
            results[group] = {"n": n, "f1": None, "skipped": "n<30"}
            continue
        results[group] = _group_metrics(
            joined["label"].values[in_group], joined["pred"].values[in_group]
        )

    return _summarize(results)


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
    for group in top_groups:
        in_group = (ko_test["grp"] == group).values
        n = int(in_group.sum())
        if n < MIN_SAMPLES:
            results[group] = {"n": n, "f1": None, "skipped": "n<30"}
            continue
        results[group] = _group_metrics(
            ko_test["label"].values[in_group], ko_test["pred"].values[in_group]
        )

    return _summarize(results)


def evaluate_disability_domain(test_df: pd.DataFrame, y_pred: np.ndarray) -> dict[str, Any]:
    """장애 키워드 포함 여부별 F1."""
    df = test_df.copy()
    df["pred"] = y_pred
    df["has_disability"] = (
        df["text"].astype(str).apply(lambda t: any(kw in t for kw in DISABILITY_KEYWORDS))
    )

    results: dict[str, Any] = {}
    for label, mask in [
        ("disability_yes", df["has_disability"]),
        ("disability_no", ~df["has_disability"]),
    ]:
        n = int(mask.sum())
        if n < MIN_SAMPLES:
            results[label] = {"n": n, "f1": None, "skipped": "n<30"}
            continue
        results[label] = _group_metrics(
            df["label"].values[mask.values], df["pred"].values[mask.values]
        )

    return _summarize(results)


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
