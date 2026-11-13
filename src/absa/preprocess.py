"""Vietnamese text normalization, word segmentation and a preprocessing cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import unicodedata
import urllib.request
from pathlib import Path
from typing import Literal

import pandas as pd
from underthesea import text_normalize, word_tokenize

PRE_DEFAULTS = {
    "lowercase": False,
    "strip_emoji": False,
    "strip_punct": False,
    "segment": None,
    "use_star": False,
}

# shortcut: text emoticons such as ":)" are not handled (see docs/decisions.md).
_EMOJI = re.compile("[\U0001f000-\U0001faff☀-➿⬀-⯿️‍⃣]")

_vncorenlp = None  # loaded lazily, once per process
# same files and URLs as py_vncorenlp.download_model (which needs wget, absent on Windows)
_VNCORENLP_BASE = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/"
_VNCORENLP_FILES = (
    "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr",
)


def _ensure_vncorenlp(save_dir: Path) -> None:
    for rel in _VNCORENLP_FILES:
        dest = save_dir / rel
        if dest.exists() and dest.stat().st_size > 0:
            continue
        made = [d for d in (dest.parent, *dest.parent.parents) if not d.exists()]
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        try:
            urllib.request.urlretrieve(_VNCORENLP_BASE + rel, part)
            part.replace(dest)
        except BaseException:
            part.unlink(missing_ok=True)
            for d in made:  # innermost first; only dirs created by this call
                try:
                    d.rmdir()
                except OSError:
                    break
            raise


def normalize(
    text: str,
    *,
    lowercase: bool = False,
    strip_emoji: bool = False,
    strip_punct: bool = False,
) -> str:
    text = text_normalize(unicodedata.normalize("NFC", text))
    if strip_emoji:
        text = _EMOJI.sub(" ", text)
    if strip_punct:
        text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    if lowercase:
        text = text.lower()
    return " ".join(text.split())


def _ensure_java_home() -> None:
    java_home = os.environ.get("JAVA_HOME")
    if java_home and Path(java_home).is_dir():
        return
    try:
        out = subprocess.run(
            ["java", "-XshowSettings:properties", "-version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        found = re.search(r"^\s*java\.home = (.+?)\s*$", out.stderr, re.M)
    except (OSError, subprocess.SubprocessError):
        found = None
    if not found:
        raise RuntimeError(
            "Không tìm thấy Java. VnCoreNLP cần Java 17 trở lên: hãy cài Java "
            "và đặt biến môi trường JAVA_HOME trỏ tới thư mục cài đặt."
        )
    os.environ["JAVA_HOME"] = found.group(1)


def _get_vncorenlp():
    global _vncorenlp
    if _vncorenlp is None:
        _ensure_java_home()
        import py_vncorenlp

        save_dir = Path("models/vncorenlp").resolve()
        cwd = os.getcwd()
        try:
            _ensure_vncorenlp(save_dir)
            _vncorenlp = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(save_dir))
        finally:
            os.chdir(cwd)  # py_vncorenlp changes the working directory
    return _vncorenlp


def segment(
    texts: list[str], backend: Literal["underthesea", "vncorenlp"] = "underthesea"
) -> list[str]:
    if backend == "underthesea":
        return [word_tokenize(t, format="text") if t.strip() else "" for t in texts]
    if backend == "vncorenlp":
        rdr = _get_vncorenlp()
        return [" ".join(rdr.word_segment(t)) if t.strip() else "" for t in texts]
    raise ValueError(f"unknown segment backend: {backend!r}")


def preprocess_df(
    df: pd.DataFrame,
    pre_cfg: dict,
    cache_dir: Path | None = None,
    cache_tag: str | None = None,
) -> list[str]:
    unknown = set(pre_cfg) - set(PRE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown preprocessing keys: {sorted(unknown)}")
    cfg = {**PRE_DEFAULTS, **pre_cfg}

    path = None
    if cache_dir is not None and cache_tag is not None:
        digest = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
        path = Path(cache_dir) / f"{cache_tag}__{digest}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached["n"] == len(df):
                return cached["texts"]

    texts = [
        normalize(
            str(t),
            lowercase=cfg["lowercase"],
            strip_emoji=cfg["strip_emoji"],
            strip_punct=cfg["strip_punct"],
        )
        for t in df["comment"]
    ]
    if cfg["segment"]:
        texts = segment(texts, cfg["segment"])
    if cfg["use_star"]:
        texts = [f"[{s} sao] {t}" for s, t in zip(df["n_star"], texts, strict=True)]

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"n": len(df), "config": cfg, "texts": texts}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return texts
