"""LightGBM 기반 메타 학습기 (Stacking) 모델 정의부."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

# 장애 키워드 명시
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


class LightGBMStacker:
    """KcELECTRA logits와 메타데이터를 결합해 최종 4단계 위험도를 분류하는 앙상블 학습기."""

    def __init__(self, lgbm_params: dict[str, Any] | None = None) -> None:
        if lgbm_params is None:
            lgbm_params = {
                "objective": "multiclass",
                "num_class": 4,
                "metric": "multi_logloss",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "random_state": 42,
                "verbose": -1,
            }
        self.params = lgbm_params
        self.model: lgb.Booster | None = None

    def build_meta_features(
        self, df: pd.DataFrame, logits: np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """logits와 텍스트 메타 피처를 결합하여 LightGBM 입력 매트릭스를 구성합니다."""
        meta_df = pd.DataFrame()

        # 1. KcELECTRA Logits (4-dim)
        if logits.shape[1] != 4:
            raise ValueError(f"Logits shape must be (N, 4), got {logits.shape}")

        for i in range(4):
            meta_df[f"logit_{i}"] = logits[:, i]

        # 2. Source 인코딩 (Categorical)
        source_map = {"unsmile": 0, "kold": 1, "synthetic_emergency_v1": 2}
        meta_df["source"] = df["source"].map(source_map).fillna(3).astype("category")

        # 3. 텍스트 길이 그룹화 (Length bucket)
        lens = df["text"].str.len().fillna(0)
        # short: < 20, med: 20-50, long: > 50
        length_bucket = pd.cut(lens, bins=[-1, 20, 50, float("inf")], labels=[0, 1, 2])
        meta_df["text_length_bucket"] = length_bucket.astype("category")

        # 4. 장애 관련 키워드 포함 여부 (Binary)
        has_disability = (
            df["text"]
            .apply(lambda t: any(kw in str(t) for kw in DISABILITY_KEYWORDS))
            .astype(np.int8)
        )
        meta_df["has_disability"] = has_disability

        # Target 추출
        targets = df["label"].values.astype(np.int8) if "label" in df.columns else np.zeros(len(df))
        return meta_df, targets

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        num_boost_round: int = 100,
    ) -> None:
        """LightGBM 모델을 훈련합니다."""
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(10)],
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측을 수행하여 클래스 확률을 반환합니다."""
        if self.model is None:
            raise RuntimeError("Model is not trained yet.")
        return self.model.predict(X)
