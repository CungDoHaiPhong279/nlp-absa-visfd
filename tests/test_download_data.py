import hashlib
import importlib.util
import zipfile
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "download_data", Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
)
dl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dl)


@pytest.fixture
def fake_zip(tmp_path: Path) -> Path:
    path = tmp_path / "src.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name in dl.FILES:
            z.writestr(name, "index,comment\n0,pin trâu\n")
    return path


def _run(monkeypatch, out: Path, zip_path: Path | None) -> int:
    def fake_urlretrieve(url, dest):
        assert zip_path is not None, "must not download when data exists"
        Path(dest).write_bytes(zip_path.read_bytes())

    monkeypatch.setattr(dl.urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setattr("sys.argv", ["download_data.py", "--data-dir", str(out)])
    return dl.main()


def test_extracts_when_checksum_matches(tmp_path, monkeypatch, fake_zip):
    monkeypatch.setattr(dl, "SHA256", hashlib.sha256(fake_zip.read_bytes()).hexdigest())
    out = tmp_path / "raw"
    assert _run(monkeypatch, out, fake_zip) == 0
    assert all((out / name).exists() for name in dl.FILES)


def test_wrong_checksum_fails_and_extracts_nothing(tmp_path, monkeypatch, fake_zip):
    out = tmp_path / "raw"
    assert _run(monkeypatch, out, fake_zip) == 1
    assert not any((out / name).exists() for name in dl.FILES)


def test_skips_download_when_files_exist(tmp_path, monkeypatch):
    out = tmp_path / "raw"
    out.mkdir()
    for name in dl.FILES:
        (out / name).write_text("x", encoding="utf-8")
    assert _run(monkeypatch, out, None) == 0
