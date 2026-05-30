"""Download seed datasets (Smilegate UnSmile, KOLD) into data/raw/.

Usage:
    python scripts/download_seed_datasets.py
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

UNSMILE = {
    "train": "https://raw.githubusercontent.com/smilegate-ai/korean_unsmile_dataset/main/unsmile_train_v1.0.tsv",
    "valid": "https://raw.githubusercontent.com/smilegate-ai/korean_unsmile_dataset/main/unsmile_valid_v1.0.tsv",
}

KOLD_URL = "https://media.githubusercontent.com/media/boychaboy/KOLD/main/data/kold_v1.json"
KOLD_SHA256 = "c11c29b972ecb7c63936cc8daa6e0925b6b891a78b6e6b3c321dbae238cf5468"
KOLD_SIZE = 32_791_434


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  -> {url}")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def fetch_unsmile() -> None:
    out_dir = RAW / "unsmile"
    for split, url in UNSMILE.items():
        dest = out_dir / f"unsmile_{split}_v1.0.tsv"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[unsmile/{split}] skip (exists, {dest.stat().st_size:,} bytes)")
            continue
        print(f"[unsmile/{split}] downloading")
        _download(url, dest)
        with dest.open(encoding="utf-8") as f:
            rows = sum(1 for _ in f) - 1
        print(f"[unsmile/{split}] ok ({dest.stat().st_size:,} bytes, {rows:,} rows)")


def fetch_kold() -> None:
    dest = RAW / "kold" / "kold_v1.json"
    if dest.exists() and dest.stat().st_size == KOLD_SIZE:
        digest = _sha256(dest)
        if digest == KOLD_SHA256:
            print("[kold] skip (exists, sha256 verified)")
            return
        print("[kold] sha256 mismatch, re-downloading")
    print("[kold] downloading (LFS)")
    _download(KOLD_URL, dest)
    digest = _sha256(dest)
    if digest != KOLD_SHA256:
        dest.unlink(missing_ok=True)
        raise SystemExit(f"[kold] sha256 mismatch: got {digest}, expected {KOLD_SHA256}")
    print(f"[kold] ok ({dest.stat().st_size:,} bytes, sha256 verified)")


def main() -> int:
    print(f"target: {RAW}")
    fetch_unsmile()
    fetch_kold()
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
