"""synthesize_emergency.py 스크립트 헬퍼·루프 가드 테스트.

scripts/는 패키지가 아니므로 파일 경로에서 모듈을 직접 로드한다.
실제 Gemini 호출은 가짜 client(stub)로 대체 → 네트워크·API 키 불필요.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "synthesize_emergency", ROOT / "scripts" / "synthesize_emergency.py"
)
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)


# ---------- _parse_jsonl ----------


def test_parse_jsonl_strips_code_fence():
    """코드펜스(```)와 빈 줄을 건너뛰고 유효 JSON만 파싱."""
    text = '```json\n{"text": "안녕"}\n\n{"text": "둘째"}\n```'
    items = se._parse_jsonl(text)
    assert len(items) == 2
    assert items[0]["text"] == "안녕"


def test_parse_jsonl_skips_invalid_lines():
    """깨진 라인은 조용히 skip."""
    text = '{"text": "ok"}\nnot json\n{"text": "ok2"}'
    items = se._parse_jsonl(text)
    assert len(items) == 2


# ---------- _extract_text ----------


def test_extract_text_normal():
    """정상 응답은 resp.text를 그대로 반환."""
    resp = SimpleNamespace(text="hello")
    assert se._extract_text(resp) == "hello"


def test_extract_text_blocked_falls_back_to_candidates():
    """resp.text 접근이 예외(차단)면 candidates.parts에서 복구."""

    class Blocked:
        @property
        def text(self):
            raise ValueError("blocked by safety")

        candidates = [
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="복구된 ")])),
            SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(text="텍스트")])),
        ]

    assert se._extract_text(Blocked()) == "복구된 텍스트"


def test_extract_text_empty_when_no_candidates():
    """차단 + 후보 없음이면 빈 문자열."""

    class Empty:
        @property
        def text(self):
            raise ValueError("blocked")

        candidates = None

    assert se._extract_text(Empty()) == ""


# ---------- _enrich ----------


def test_enrich_non_boundary_forces_label_3():
    """3a~3d는 무조건 긴급(3) 라벨."""
    items = [{"text": "테스트 메시지"}]
    out = se._enrich(items, "3a", "train")
    assert len(out) == 1
    assert out[0]["label"] == 3
    assert out[0]["source"] == "synthetic_emergency_v1"
    assert out[0]["subcategory"] == "3a"


def test_enrich_boundary_keeps_label():
    """boundary는 0/1/2 라벨 유지, 범위 밖이면 drop."""
    items = [
        {"text": "정상 발화", "label": 0},
        {"text": "긴급 라벨은 boundary에 부적합", "label": 3},
    ]
    out = se._enrich(items, "boundary", "val")
    assert len(out) == 1
    assert out[0]["label"] == 0


def test_enrich_drops_empty_or_too_long():
    """빈 텍스트·500자 초과·text 키 없음은 제외."""
    items = [
        {"text": ""},
        {"text": "가" * 501},
        {"persona": "no text key"},
        {"text": "정상"},
    ]
    out = se._enrich(items, "3b", "test")
    assert len(out) == 1
    assert out[0]["text"] == "정상"


# ---------- 무한 루프 가드 ----------


class _StubClient:
    """generate_content가 항상 빈/차단 응답을 주는 stub (무한 루프 유발 조건)."""

    def __init__(self):
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            text="",
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=0),
            candidates=[],
        )


def test_empty_streak_guard_stops(monkeypatch, tmp_path):
    """모든 배치가 비어도 MAX_EMPTY_STREAK에서 멈추고 quota를 무한 소진하지 않음."""
    monkeypatch.setattr(se, "ROOT", tmp_path)  # 최종 print의 relative_to용
    monkeypatch.setattr(se, "OUT_ROOT", tmp_path / "emergency")
    monkeypatch.setattr(se, "RATE_LIMIT_SLEEP", 0)  # 테스트 가속
    monkeypatch.setattr(se, "TARGET_COUNTS", {"3a": {"train": 100}})

    stub = _StubClient()

    # synthesize 내부의 `from google import genai` 를 stub로 대체
    class _FakeGenai:
        @staticmethod
        def Client(api_key):  # noqa: N802  # genai SDK 시그니처 모사
            return stub

    monkeypatch.setitem(sys.modules, "google.genai", _FakeGenai)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    summary = se.synthesize(["3a"], dry_run=False)

    # MAX_EMPTY_STREAK회만 호출하고 중단 (무한 루프 아님)
    assert stub.calls == se.MAX_EMPTY_STREAK
    assert summary["3a"]["train"] == 0
