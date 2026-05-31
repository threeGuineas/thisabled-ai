"""모듈 2 (LambdaMART) 매칭 랭킹을 위한 쿼리-후보 페어 생성 로직."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def generate_mock_profiles(n: int = 1000) -> pd.DataFrame:
    """스키마 명세서 기준 사용자 프로필 1,000건 랜덤 생성."""
    np.random.seed(42)
    regions = ["서울", "경기", "부산", "대구", "인천"]
    disabilities = ["발달장애", "시각장애", "청각장애", "지체장애", "비장애"]
    interests_pool = ["운동", "음악", "독서", "게임", "맛집", "영화", "여행", "사진", "요리"]

    profiles = []
    for i in range(n):
        age = int(np.random.normal(30, 8))
        age = max(15, min(70, age))
        region = np.random.choice(regions)
        disability = np.random.choice(disabilities)
        n_inter = np.random.randint(1, 5)
        interests = np.random.choice(interests_pool, n_inter, replace=False).tolist()

        intro = f"안녕하세요. 저는 {region}에 사는 {age}세 {disability}인입니다. "
        intro += " ".join(interests) + " 좋아해요. 반가워요!"

        profiles.append(
            {
                "user_id": f"usr_{i}",
                "introduction": intro,
                "age": age,
                "region": region,
                "disability_type": disability,
                "interests": interests,
            }
        )
    return pd.DataFrame(profiles)


def build_pairs(
    df_profiles: pd.DataFrame, n_queries: int = 200, n_candidates: int = 15
) -> pd.DataFrame:
    """LambdaMART 훈련용 주 사용자(Query) - 대상 사용자(Cand) 페어 생성 및 라벨 합성."""
    np.random.seed(42)
    users = df_profiles.to_dict("records")

    pairs = []
    queries = np.random.choice(users, n_queries, replace=False)
    for q in queries:
        candidates = np.random.choice(users, n_candidates, replace=False)
        for c in candidates:
            if q["user_id"] == c["user_id"]:
                continue

            # 메타데이터 추출
            age_diff = abs(q["age"] - c["age"])
            region_match = int(q["region"] == c["region"])
            dis_match = int(q["disability_type"] == c["disability_type"])
            overlap = len(set(q["interests"]).intersection(set(c["interests"])))

            # Rule-based Relevance (0 ~ 3) - 명세서 참고
            label = 0
            if region_match and age_diff <= 5 and overlap >= 2:
                label = 3
            elif region_match and age_diff <= 10 and overlap >= 1:
                label = 2
            elif region_match or overlap >= 1:
                label = 1

            pairs.append(
                {
                    "query_id": q["user_id"],
                    "cand_id": c["user_id"],
                    "age_diff": age_diff,
                    "region_match": region_match,
                    "dis_match": dis_match,
                    "overlap": overlap,
                    "label": label,
                    "q_intro": q["introduction"],
                    "c_intro": c["introduction"],
                }
            )

    df_pairs = pd.DataFrame(pairs)
    # 쿼리 그룹 단위로 정렬 (LightGBM 요구사항)
    df_pairs = df_pairs.sort_values(by="query_id").reset_index(drop=True)
    return df_pairs


def get_sbert_embeddings(
    texts: list[str], model_name: str = "jhgan/ko-sroberta-multitask"
) -> np.ndarray:
    """Ko-sroberta-multitask 모델로 임베딩 추출."""
    model = SentenceTransformer(model_name)
    return model.encode(texts, batch_size=64, show_progress_bar=True)


def engineer_features(df_pairs: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    """SBERT 임베딩 유사도 및 메타데이터 피처화."""
    print("  - SBERT 텍스트 임베딩 추출 중...")
    q_texts = df_pairs["q_intro"].tolist()
    c_texts = df_pairs["c_intro"].tolist()

    q_embs = get_sbert_embeddings(q_texts)
    c_embs = get_sbert_embeddings(c_texts)

    # 4.1 SBERT 임베딩 피처
    cosine = np.sum(q_embs * c_embs, axis=1) / (
        np.linalg.norm(q_embs, axis=1) * np.linalg.norm(c_embs, axis=1) + 1e-8
    )
    l2 = np.linalg.norm(q_embs - c_embs, axis=1)

    # 4.2 최종 피처 프레임워크
    features = pd.DataFrame(
        {
            "f_cosine": cosine.astype(np.float32),
            "f_l2": l2.astype(np.float32),
            "f_age_diff": df_pairs["age_diff"].astype(np.float32),
            "f_region_match": df_pairs["region_match"].astype(np.float32),
            "f_dis_match": df_pairs["dis_match"].astype(np.float32),
            "f_overlap": df_pairs["overlap"].astype(np.float32),
        }
    )

    # LightGBM group 파라미터 계산
    groups = df_pairs.groupby("query_id").size().tolist()

    return features, groups
