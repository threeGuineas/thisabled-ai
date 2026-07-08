"""SAFE 서빙 /health 계약 — revision 보고·pin 배선 테스트.

실모델 다운로드 없이 transformers 로더를 stub으로 대체해 배선만 검증한다:
- SAFE_MODEL_REVISION 환경변수가 from_pretrained(revision=...)로 전달되는가
- /health 가 실제 로드된 커밋 SHA(config._commit_hash)를 revision 으로 보고하는가
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient

# serving/ 를 sys.path 에 추가해 safety_server 패키지를 import 가능하게 함
_SERVING = Path(__file__).resolve().parents[1] / "serving"
if str(_SERVING) not in sys.path:
    sys.path.insert(0, str(_SERVING))

RESOLVED_SHA = "79bbd16e2ea9a5c9133fb01c6f8c1c09671283aa"


class _StubConfig:
    def __init__(self) -> None:
        self.num_labels = 2
        self._commit_hash = RESOLVED_SHA


class _StubModel:
    def __init__(self) -> None:
        self.config = _StubConfig()

    def eval(self) -> _StubModel:
        return self

    def __call__(self, **enc):
        class _Out:
            logits = torch.tensor([[2.0, -2.0]])

        return _Out()


class _StubTokenizer:
    def __call__(self, text, **kw):
        return {"input_ids": torch.tensor([[1, 2, 3]])}


def _load_app(monkeypatch, revision: str | None):
    if revision is None:
        monkeypatch.delenv("SAFE_MODEL_REVISION", raising=False)
    else:
        monkeypatch.setenv("SAFE_MODEL_REVISION", revision)
    monkeypatch.setenv("SAFE_MODEL_DIR", "soyuncj/thisabled-safety-kcelectra")

    sys.modules.pop("safety_server.app", None)
    app_mod = importlib.import_module("safety_server.app")

    calls = {}

    def _tok_from_pretrained(model_dir, revision=None, **kw):
        calls["tok_revision"] = revision
        return _StubTokenizer()

    def _model_from_pretrained(model_dir, revision=None, **kw):
        calls["model_revision"] = revision
        return _StubModel()

    monkeypatch.setattr(app_mod.AutoTokenizer, "from_pretrained", _tok_from_pretrained)
    monkeypatch.setattr(
        app_mod.AutoModelForSequenceClassification, "from_pretrained", _model_from_pretrained
    )
    return app_mod, calls


def test_health_reports_resolved_revision(monkeypatch):
    app_mod, calls = _load_app(monkeypatch, RESOLVED_SHA)
    with TestClient(app_mod.app) as client:
        body = client.get("/health").json()
    # 요청 revision 이 두 로더 모두에 전달됐는지
    assert calls["tok_revision"] == RESOLVED_SHA
    assert calls["model_revision"] == RESOLVED_SHA
    # /health 가 실제 로드된 커밋을 보고
    assert body["status"] == "ok"
    assert body["loaded"] is True
    assert body["revision"] == RESOLVED_SHA
    assert body["num_labels"] == 2
    assert body["labels"] == ["정상", "주의"]


def test_no_revision_defaults_to_none(monkeypatch):
    app_mod, calls = _load_app(monkeypatch, None)
    with TestClient(app_mod.app) as client:
        body = client.get("/health").json()
    # 미지정이면 None(=main HEAD) 으로 전달, /health 는 stub 이 해석한 실제 커밋을 보고
    assert calls["model_revision"] is None
    assert body["revision"] == RESOLVED_SHA
