"""Tests for scripts/train_stacking.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.train_stacking import build_meta_features


def test_build_meta_features_shape_and_types():
    # 3건의 더미 텍스트 DataFrame 생성
    df = pd.DataFrame(
        {
            "text": [
                "안녕하세요. 반갑습니다.",
                "발달장애 청소년을 위한 쉼터입니다.",
                "너 진짜 죽고 싶냐? 오늘 학교 뒤로 와라.",
            ],
            "label": [0, 1, 3],
            "source": ["unsmile", "kold", "synthetic_emergency_v1"],
        }
    )

    # 3건에 대한 4차원 dummy logits (임의의 float)
    logits = np.array(
        [
            [1.5, -0.5, -1.0, -2.0],
            [-0.5, 2.0, 0.5, -1.5],
            [-2.0, -1.0, 1.0, 3.5],
        ],
        dtype=np.float32,
    )

    # 메타 피처 빌드 실행
    X, y = build_meta_features(df, logits)  # noqa: N806

    # shape 검증
    assert X.shape == (3, 7)  # logit_0~3 (4) + source (1) + length (1) + has_disability (1)
    assert len(y) == 3

    # target 값 일치 검증
    assert np.array_equal(y, [0, 1, 3])

    # 컬럼 존재 여부
    expected_cols = [
        "logit_0",
        "logit_1",
        "logit_2",
        "logit_3",
        "source",
        "length",
        "has_disability",
    ]
    assert list(X.columns) == expected_cols

    # source 인코딩 확인 (Categorical)
    assert X["source"].dtype.name == "category"
    assert list(X["source"]) == [0, 1, 2]  # unsmile=0, kold=1, synthetic=2

    # length 검증 (글자 수)
    # "안녕하세요. 반갑습니다." -> 13자
    # "발달장애 청소년을 위한 쉼터입니다." -> 19자
    # "너 진짜 죽고 싶냐? 오늘 학교 뒤로 와라." -> 24자
    assert list(X["length"]) == [13.0, 19.0, 24.0]

    # 장애 키워드 필터링 검증
    # 첫번째: 없음 (0)
    # 두번째: "발달장애" 포함 (1)
    # 세번째: 없음 (0)
    assert list(X["has_disability"]) == [0, 1, 0]
