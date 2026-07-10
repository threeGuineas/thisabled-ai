from __future__ import annotations

import importlib.util
import io
from pathlib import Path

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
