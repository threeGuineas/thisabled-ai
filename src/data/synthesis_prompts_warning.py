"""경고(2) 합성 데이터용 프롬프트 및 메타데이터 정의.

AI-Hub 기준 경고(2)의 정의 (intensity >= 2.0 AND (HATE|DISCRIMINATION|CENSURE|ABUSE))를 충족하면서,
시드 데이터(UnSmile/KOLD)와 AI-Hub 평가셋 간의 라벨 정의 차이를 좁히기 위한 합성용 명세.
"""

from __future__ import annotations

# 카테고리별 목표 건수 (Optuna 탐색 시 최대 5배수 증강을 위해 충분한 풀 생성)
# 650건 기준 5배 증강 = 3250건이 최대치이므로, 풀을 약 3000~4000건 확보해둔다.
TARGET_COUNTS = {
    "2a": {"train": 1500, "val": 0, "test": 0},  # 차별/혐오
    "2b": {"train": 1500, "val": 0, "test": 0},  # 강한 비난/모욕
    "boundary": {"train": 500, "val": 0, "test": 0},  # 모호한 주의(1) 등 반례
}
