"""Tải UIT-ViSFD, kiểm tra SHA-256 và giải nén vào data/raw."""

import argparse
import hashlib
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://raw.githubusercontent.com/LuongPhan/UIT-ViSFD/main/UIT-ViSFD.zip"
SHA256 = "d3c068814d23b11473c6bf774a81ffe1916b304501ac10e0c99fcd15f11a16a8"
FILES = ("Train.csv", "Dev.csv", "Test.csv")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1258 không in được tiếng Việt
    sys.stderr.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    out = ap.parse_args().data_dir
    if all((out / f).exists() for f in FILES):
        print("Đã có dữ liệu")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(URL, tmp_path)
        digest = hashlib.sha256(tmp_path.read_bytes()).hexdigest()
        if digest != SHA256:
            print(f"SHA-256 sai: {digest} (cần {SHA256})", file=sys.stderr)
            return 1
        with zipfile.ZipFile(tmp_path) as z:
            z.extractall(out)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(f"Đã tải vào {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
