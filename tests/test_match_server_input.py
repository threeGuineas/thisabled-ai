"""MATCH `/score`의 확장 입력 계약과 개인정보 비노출 테스트."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from serving.match_server.app import INPUT_POLICY, _find_default_config_path, _state, app
from src.data.matching_input import ALLOWED_RECOMMENDATION_REASONS

NOW_ISO = datetime.now(UTC).isoformat()


class FakeEncoder:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.seen.extend(sentences)
        return np.asarray([[1.0, 0.0, 0.0] for _ in sentences], dtype=np.float32)


class FakeRanker:
    def __init__(self) -> None:
        self.seen_columns: list[str] = []
        self.call_count = 0

    def predict(self, features):
        self.call_count += 1
        self.seen_columns = list(features.columns)
        return np.zeros(len(features), dtype=np.float32)


@pytest.fixture
def client_and_models():
    encoder = FakeEncoder()
    ranker = FakeRanker()
    _state.clear()
    _state.update({"sbert": encoder, "ranker": ranker})
    client = TestClient(app, raise_server_exceptions=False)
    yield client, encoder, ranker
    client.close()
    _state.clear()


def legacy_user(user_id: str, **overrides) -> dict:
    payload = {
        "user_id": user_id,
        "bio": "영화 이야기를 좋아해요.",
        "tags": ["movie"],
        "age_band": "25-34",
        "ui_mode": "visual",
    }
    payload.update(overrides)
    return payload


def test_validation_error_does_not_echo_forbidden_input(client_and_models) -> None:
    client, _encoder, _ranker = client_and_models
    canary = "1990-01-01-secret@example.com"
    body = {
        "me": {**legacy_user("me"), "birth_date": canary},
        "candidates": [],
    }

    response = client.post("/score", json=body)

    assert response.status_code == 422
    assert canary not in response.text
    assert "birth_date" in response.text


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "nickname",
        "profile_image_url",
        "chat_messages",
        "disability_type",
        "risk_classification_history",
        "exact_location",
    ],
)
def test_forbidden_profile_fields_are_rejected_without_echo(
    client_and_models, forbidden_field: str
) -> None:
    client, _encoder, _ranker = client_and_models
    canary = "private-canary@example.com"
    response = client.post(
        "/score",
        json={
            "me": {**legacy_user("me"), forbidden_field: canary},
            "candidates": [],
        },
    )
    assert response.status_code == 422
    assert forbidden_field in response.text
    assert canary not in response.text


@pytest.mark.parametrize(
    "bio",
    [
        "연락처 010-1234-5678",
        "메일 secret@example.com",
        "카톡 아이디: private_id",
        "텔레그램은 secret123",
    ],
)
def test_contact_validation_returns_code_without_value(client_and_models, bio: str) -> None:
    client, encoder, ranker = client_and_models
    body = {
        "me": legacy_user("me", bio=bio),
        "candidates": [legacy_user("cand")],
    }

    response = client.post("/score", json=body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CONTACT_INFO_DETECTED"
    assert bio not in response.text
    assert encoder.seen == []
    assert ranker.call_count == 0


def test_contact_is_rejected_even_when_candidate_list_is_empty(client_and_models) -> None:
    client, encoder, ranker = client_and_models
    bio = "연락처 010-9876-5432"
    response = client.post(
        "/score",
        json={"me": legacy_user("me", bio=bio), "candidates": []},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CONTACT_INFO_DETECTED"
    assert bio not in response.text
    assert encoder.seen == []
    assert ranker.call_count == 0


def test_missing_age_is_rejected_fail_closed(client_and_models) -> None:
    client, encoder, ranker = client_and_models
    response = client.post(
        "/score",
        json={
            "me": legacy_user("me", age_band=""),
            "candidates": [legacy_user("cand")],
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "MISSING_AGE"
    assert encoder.seen == []
    assert ranker.call_count == 0


def test_server_enforces_configured_tag_01_registry(client_and_models) -> None:
    client, encoder, ranker = client_and_models
    assert {"movie", "walking", "healing"}.issubset(INPUT_POLICY.allowed_tag_ids or set())
    response = client.post(
        "/score",
        json={
            "me": legacy_user("me", tags=["not-a-tag-01-code"]),
            "candidates": [legacy_user("cand")],
        },
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "UNKNOWN_TAG"
    assert encoder.seen == []
    assert ranker.call_count == 0


def test_default_config_path_falls_back_to_repository(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = _find_default_config_path()

    assert config_path.is_file()
    assert config_path.name == "module2_matching.yaml"
    assert config_path.parent.name == "configs"


def test_default_config_path_supports_flat_docker_layout(tmp_path) -> None:
    service_root = tmp_path / "srv"
    config_path = service_root / "configs" / "module2_matching.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("input: {}\n", encoding="utf-8")

    found = _find_default_config_path(
        cwd=tmp_path / "different-cwd",
        module_file=service_root / "app.py",
    )

    assert found == config_path


def test_legacy_payload_remains_compatible_and_response_is_minimal(client_and_models) -> None:
    client, _encoder, ranker = client_and_models
    response = client.post(
        "/score",
        json={"me": legacy_user("me"), "candidates": [legacy_user("cand")]},
    )

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 1
    assert set(results[0]) == {"user_id", "score", "reasons"}
    assert 0.0 <= results[0]["score"] <= 1.0
    assert ranker.seen_columns == ["f_cosine", "f_l2", "f_dis_match"]
    assert set(results[0]["reasons"]).issubset(ALLOWED_RECOMMENDATION_REASONS)


def test_relationship_filtered_candidate_never_reaches_encoder_or_ranker(
    client_and_models,
) -> None:
    client, encoder, ranker = client_and_models
    canary_bio = "blocked-bio-canary@example.com"
    canary_post = "blocked-post-canary@example.com"
    blocked = legacy_user(
        "blocked",
        bio=canary_bio,
        authored_items=[
            {
                "content_id": "post-1",
                "source_type": "post",
                "text": canary_post,
                "created_at": NOW_ISO,
            }
        ],
        liked_items=[
            {
                "content_id": "like-1",
                "source_type": "comment",
                "text": "blocked-like-canary@example.com",
                "created_at": NOW_ISO,
            }
        ],
        relationship={"blocked_either_direction": True},
    )

    response = client.post(
        "/score",
        json={"me": legacy_user("me"), "candidates": [blocked]},
    )

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert canary_bio not in encoder.seen
    assert canary_post not in encoder.seen
    assert "blocked-like-canary@example.com" not in encoder.seen
    assert ranker.call_count == 0


def test_minor_and_adult_are_filtered_even_for_legacy_payload(client_and_models) -> None:
    client, _encoder, ranker = client_and_models
    response = client.post(
        "/score",
        json={
            "me": legacy_user("adult", age_band="19-24"),
            "candidates": [legacy_user("minor", age_band="14-18")],
        },
    )

    assert response.status_code == 200
    assert response.json()["results"] == []
    assert ranker.call_count == 0


def test_relationship_boolean_strings_are_rejected_without_echo(client_and_models) -> None:
    client, encoder, ranker = client_and_models
    response = client.post(
        "/score",
        json={
            "me": legacy_user("me"),
            "candidates": [
                legacy_user(
                    "cand",
                    relationship={"blocked_either_direction": "false-secret-canary"},
                )
            ],
        },
    )
    assert response.status_code == 422
    assert "false-secret-canary" not in response.text
    assert encoder.seen == []
    assert ranker.call_count == 0


def test_authored_liked_and_relationship_inputs_reach_generalized_features(
    client_and_models,
) -> None:
    client, encoder, ranker = client_and_models
    now = NOW_ISO
    me = legacy_user(
        "me",
        authored_items=[
            {"content_id": "p1", "source_type": "post", "text": "영화 후기", "created_at": now}
        ],
        liked_items=[
            {"content_id": "l1", "source_type": "comment", "text": "야구 이야기", "created_at": now}
        ],
    )
    candidate = legacy_user(
        "cand",
        authored_items=[
            {"content_id": "p2", "source_type": "post", "text": "영화 감상", "created_at": now}
        ],
        liked_items=[
            {"content_id": "l2", "source_type": "comment", "text": "야구 소식", "created_at": now}
        ],
        relationship={"common_friend_count": 2},
    )

    response = client.post("/score", json={"me": me, "candidates": [candidate]})

    assert response.status_code == 200
    item = response.json()["results"][0]
    assert "영화 후기" in encoder.seen
    assert "야구 이야기" in encoder.seen
    assert "관심 있는 콘텐츠가 비슷해요" in item["reasons"]
    assert "공통 친구가 있어요" in item["reasons"]
    assert ranker.call_count == 1


def test_no_text_tag_ui_or_common_friend_returns_insufficient_signal(
    client_and_models,
) -> None:
    client, encoder, ranker = client_and_models
    empty = {"bio": "", "tags": [], "age_band": "25-34", "ui_mode": ""}
    response = client.post(
        "/score",
        json={
            "me": {"user_id": "me", **empty},
            "candidates": [{"user_id": "cand", **empty}],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"results": [], "message": "추천 정보가 부족합니다"}
    assert encoder.seen == []
    assert ranker.call_count == 0
