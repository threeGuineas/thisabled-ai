from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "adapt_external_datasets.py"
SPEC = importlib.util.spec_from_file_location("adapt_external_datasets", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ensure_dktc_csv_downloads_and_validates_schema(tmp_path, monkeypatch):
    target = tmp_path / "dktc" / "train.csv"
    payload = "idx,class,conversation\n1,threat,대화 내용\n".encode()

    monkeypatch.setattr(MODULE, "DKTC_CSV", target)
    monkeypatch.setattr(
        MODULE.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload)
    )

    result = MODULE.ensure_dktc_csv()

    assert result == target
    assert target.read_bytes() == payload


def test_exact_normalizer_preserves_jamo_and_ignores_spacing_punctuation():
    assert MODULE._norm("ㅋ ㅋ ㅋ ㅋ!") == "ㅋㅋㅋㅋ"
    assert MODULE._norm("삼가 고인의 명복을 빕니다...") == MODULE._norm("삼가고인의명복을빕니다")


def test_load_forbidden_uses_only_consumed_blind_v1_to_v9(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    fixture_dir = tmp_path / "fixtures"
    grooming_dir = tmp_path / "grooming"
    eval_dir.mkdir()
    fixture_dir.mkdir()
    grooming_dir.mkdir()
    for name in ("aihub_real_holdout.jsonl", "beep_real_holdout.jsonl"):
        (eval_dir / name).write_text(
            json.dumps({"text": name}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    for version in (1, 9, 10):
        (fixture_dir / f"safe_blind_v{version}.jsonl").write_text(
            json.dumps({"text": f"blind-{version}"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    grooming_files = []
    for split in ("val", "test"):
        path = grooming_dir / f"{split}.jsonl"
        path.write_text(
            json.dumps({"text": f"groom-{split}"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        grooming_files.append(path)

    monkeypatch.setattr(MODULE, "EVAL_DIR", eval_dir)
    monkeypatch.setattr(MODULE, "FIXTURE_DIR", fixture_dir)
    monkeypatch.setattr(MODULE, "GROOMING_DEV_FILES", tuple(grooming_files))

    texts = set(MODULE.load_forbidden_texts())

    assert "blind-1" in texts
    assert "blind-9" in texts
    assert "blind-10" not in texts
    assert {"groom-val", "groom-test"} <= texts


def test_dedup_report_removes_punctuation_variant_before_minhash(monkeypatch):
    frame = pd.DataFrame(
        {
            "text": ["삼가고인의명복을빕니다...", "서로 다른 문장"],
            "label": [0, 1],
        }
    )
    forbidden = pd.Series(["삼가 고인의 명복을 빕니다"])
    monkeypatch.setattr(
        MODULE,
        "deduplicate_against",
        lambda _forbidden, candidate, **_kwargs: (candidate, 0),
    )

    result = MODULE.dedup_report(frame, forbidden, "test")

    assert result["text"].tolist() == ["서로 다른 문장"]
