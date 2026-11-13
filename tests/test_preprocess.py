import os
import shutil
import subprocess
import unicodedata

import pandas as pd
import pytest

from absa import preprocess as P


def test_nfc_and_tone_consistent():
    out = P.normalize(unicodedata.normalize("NFD", "Pin hóa đá"))
    assert out == unicodedata.normalize("NFC", out)
    assert P.normalize("hoá đá") == P.normalize("hóa đá")


def test_strip_emoji():
    assert P.normalize("pin trâu 😍👍", strip_emoji=True) == "pin trâu"
    assert P.normalize("😍😍", strip_emoji=True) == ""
    assert P.normalize("pin trâu 😍").endswith("😍")  # mặc định giữ emoji


def test_strip_punct_lower_whitespace():
    assert P.normalize("pin trâu!!! máy ok...", strip_punct=True) == "pin trâu máy ok"
    assert P.normalize("Pin TRÂU", lowercase=True) == "pin trâu"
    assert P.normalize("  pin \n trâu ") == "pin trâu"
    assert P.normalize("") == ""


def test_segment_underthesea():
    out = P.segment(["điện thoại pin trâu", ""])
    assert "điện_thoại" in out[0] and out[1] == ""


@pytest.mark.skipif(shutil.which("java") is None, reason="không có Java")
def test_vncorenlp_keeps_cwd():
    cwd = os.getcwd()
    out = P.segment(["điện thoại pin trâu"], backend="vncorenlp")
    assert "điện_thoại" in out[0] and os.getcwd() == cwd


def test_use_star_prefix_after_segment():
    df = pd.DataFrame({"comment": ["điện thoại tốt"], "n_star": [5]})
    out = P.preprocess_df(df, {"segment": "underthesea", "use_star": True})
    assert out[0].startswith("[5 sao] ") and "điện_thoại" in out[0]


def test_unknown_key_raises():
    with pytest.raises(ValueError, match="lowercse"):
        P.preprocess_df(pd.DataFrame({"comment": ["a"], "n_star": [1]}), {"lowercse": True})


def test_cache_hit(tmp_path, monkeypatch):
    df = pd.DataFrame({"comment": ["pin trâu", "máy đẹp"], "n_star": [5, 4]})
    first = P.preprocess_df(df, {"segment": "underthesea"}, tmp_path, "train")
    monkeypatch.setattr(
        P, "segment", lambda *a, **k: (_ for _ in ()).throw(AssertionError("không dùng cache"))
    )
    assert P.preprocess_df(df, {"segment": "underthesea"}, tmp_path, "train") == first
    assert len(list(tmp_path.glob("train__*.json"))) == 1


def test_ensure_vncorenlp_cleans_up_on_failure(tmp_path, monkeypatch):
    calls = []

    def fake(url, dest):
        calls.append(dest)
        with open(dest, "wb") as f:
            f.write(b"x")
        if len(calls) == 2:
            raise OSError("network down")

    monkeypatch.setattr(P.urllib.request, "urlretrieve", fake)
    with pytest.raises(OSError, match="network down"):
        P._ensure_vncorenlp(tmp_path)
    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert [p.name for p in files] == ["VnCoreNLP-1.2.jar"]  # first file complete, no .part
    assert all(p.stat().st_size > 0 for p in files)


def test_ensure_java_home_from_java_settings(monkeypatch):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    err = "settings:\n    java.home = " + r"C:\fake\jdk" + "\n"
    fake = subprocess.CompletedProcess([], 0, stdout="", stderr=err)
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: fake)
    P._ensure_java_home()
    assert os.environ["JAVA_HOME"] == r"C:\fake\jdk"
    monkeypatch.delenv("JAVA_HOME")
    empty = subprocess.CompletedProcess([], 0, "", "")
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: empty)
    with pytest.raises(RuntimeError, match="JAVA_HOME"):
        P._ensure_java_home()


def test_ensure_vncorenlp_leaves_no_empty_dirs(tmp_path, monkeypatch):
    def fail(url, dest):
        raise OSError("network down")

    monkeypatch.setattr(P.urllib.request, "urlretrieve", fail)
    save_dir = tmp_path / "vncorenlp"
    with pytest.raises(OSError, match="network down"):
        P._ensure_vncorenlp(save_dir)
    assert not save_dir.exists()
